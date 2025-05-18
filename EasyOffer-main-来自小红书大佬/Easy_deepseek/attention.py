"""
多层注意力机制模块 - 实现高效的多头注意力层。
该模块提供优化的注意力计算，支持KV缓存、旋转位置编码和低秩适配技术。
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from config import ModelArgs, attn_impl
from linear_layers import Linear, ColumnParallelLinear, RowParallelLinear, RMSNorm
from distributed import get_distributed_info
from rotary_embeddings import apply_rotary_emb
from quantization import weight_dequant


class MLA(nn.Module):
    """
    多层注意力(Multi-Layer Attention)机制实现。
    
    该实现包含多项关键优化：
    1. 查询/键/值投影使用低秩适配(LoRA)以减少参数量
    2. 将查询/键分为两部分：使用和不使用旋转位置编码
    3. 支持两种不同的注意力计算实现方式(naive和absorb)
    4. 针对超长序列的优化和缩放调整
    
    属性:
        dim (int): 输入特征维度
        n_heads (int): 总注意力头数
        n_local_heads (int): 每个进程计算的注意力头数
        q_lora_rank (int): 查询投影的低秩适配维度
        kv_lora_rank (int): 键值投影的低秩适配维度
        qk_nope_head_dim (int): 非位置编码的查询/键头维度
        qk_rope_head_dim (int): 旋转位置编码的查询/键头维度
        qk_head_dim (int): 总查询/键头维度
        v_head_dim (int): 值头维度
        softmax_scale (float): 注意力分数的缩放因子
    """
    def __init__(self, args: ModelArgs):
        """
        初始化多层注意力机制。
        
        参数:
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        world_size, _ = get_distributed_info()
        
        # 基本设置
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # 查询投影：普通投影或低秩投影
        if self.q_lora_rank == 0:
            # 直接投影方式
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 低秩适配投影方式
            self.wq_a = Linear(self.dim, self.q_lora_rank)  # 降维
            self.q_norm = RMSNorm(self.q_lora_rank)         # 中间归一化
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # 升维
        
        # 键值投影：始终使用低秩投影
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, 
                                           self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # 输出投影
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        
        # 注意力分数缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        # 对超长序列进行额外缩放
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 根据注意力实现方式初始化不同的缓存
        if attn_impl == "naive":
            # 朴素实现：分别缓存K和V
            self.register_buffer("k_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim
            ), persistent=False)
            self.register_buffer("v_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim
            ), persistent=False)
        else:
            # 优化实现：缓存中间结果和位置编码
            self.register_buffer("kv_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, self.kv_lora_rank
            ), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim
            ), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        """
        多层注意力机制的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim]
            start_pos (int): 序列起始位置，用于KV缓存
            freqs_cis (torch.Tensor): 预计算的旋转位置编码
            mask (torch.Tensor, optional): 注意力掩码，用于自回归生成
            
        返回:
            torch.Tensor: 注意力层输出 [batch_size, seq_len, dim]
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # 计算查询(Q)
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # 重塑查询维度并分离两部分
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转位置编码
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # 计算键值(KV)的第一阶段
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # 根据实现方式执行不同的注意力计算
        if attn_impl == "naive":
            # 朴素实现：标准注意力计算
            
            # 合并查询的两部分
            q = torch.cat([q_nope, q_pe], dim=-1)
            
            # 计算键值向量
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            
            # 合并键的两部分
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            
            # 更新KV缓存
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            
            # 计算注意力分数
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 优化实现：融合计算减少内存访问
            
            # 获取并可能反量化权重
            from config import block_size
            wkv_b = self.wkv_b.weight 
            if hasattr(self.wkv_b, "scale") and self.wkv_b.scale is not None:
                wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            
            # 重塑权重以匹配注意力头
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            
            # 预计算查询与权重的点积
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # 更新缓存
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            
            # 计算两部分注意力分数并合并
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        
        # 应用注意力掩码
        if mask is not None:
            scores += mask.unsqueeze(1)  # 掩码扩展维度以适应注意力头
        
        # 计算注意力权重
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # 计算加权值向量
        if attn_impl == "naive":
            # 朴素实现：使用完整缓存的V
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # 优化实现：使用低秩表示计算
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        # 合并多头注意力并通过输出投影
        x = self.wo(x.flatten(2))
        
        return x