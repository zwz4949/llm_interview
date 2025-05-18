"""
Transformer模型模块 - 组合所有组件构建完整的Transformer架构。
该模块整合嵌入层、Transformer块和输出层，提供完整的模型前向传播逻辑。
"""

import torch
from torch import nn
import torch.distributed as dist
from typing import Optional

from config import ModelArgs
from distributed import get_distributed_info, initialize_distributed
from linear_layers import ParallelEmbedding, ColumnParallelLinear, RMSNorm
from rotary_embeddings import precompute_freqs_cis
from transformer_block import Block


class Transformer(nn.Module):
    """
    完整的Transformer模型实现，整合各组件。
    
    该模型实现了自回归语言模型架构，具有以下特点：
    1. 使用词嵌入表示输入token
    2. 多层Transformer块处理序列
    3. 旋转位置编码提供位置信息
    4. 支持模型并行训练和推理
    5. 优化的KV缓存机制加速自回归生成
    6. 混合精度和量化支持
    
    属性:
        max_seq_len (int): 模型支持的最大序列长度
        embed (nn.Module): 词嵌入层
        layers (nn.ModuleList): Transformer块列表
        norm (nn.Module): 最终输出归一化
        head (nn.Module): 输出投影层(logits头)
        freqs_cis (torch.Tensor): 预计算的旋转位置编码
    """
    def __init__(self, args: ModelArgs):
        """
        初始化Transformer模型。
        
        参数:
            args (ModelArgs): 模型配置参数
        """
        # 确保分布式环境已初始化
        initialize_distributed()
        world_size, rank = get_distributed_info()
        
        # 设置线性层的默认数据类型
        from linear_layers import Linear
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        
        super().__init__()
        self.max_seq_len = args.max_seq_len
        
        # 初始化词嵌入层(使用并行嵌入以支持大词汇量)
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        
        # 创建多层Transformer块
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        
        # 最终层归一化
        self.norm = RMSNorm(args.dim)
        
        # 输出投影层(logits头)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        
        # 预计算旋转位置编码
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Transformer模型的前向传播(推理模式)。
        
        处理输入token序列，生成下一个token的预测分布。
        
        参数:
            tokens (torch.Tensor): 输入token的ID，形状为 [batch_size, seq_len]
            start_pos (int, optional): 序列起始位置，用于KV缓存，默认为0
            
        返回:
            torch.Tensor: 下一个token的logits，形状为 [batch_size, vocab_size]
        """
        # 获取序列长度
        seqlen = tokens.size(1)
        
        # 将token ID转换为嵌入表示
        h = self.embed(tokens)
        
        # 获取当前序列位置的旋转编码
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        
        # 为自回归生成创建注意力掩码(仅当序列长度大于1时需要)
        mask = None
        if seqlen > 1:
            # 创建上三角掩码，确保当前位置只能看到之前的token
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        
        # 依次通过各Transformer块
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        # 应用最终归一化
        h = self.norm(h)
        
        # 仅使用序列最后位置的输出进行预测
        h = h[:, -1]
        
        # 生成logits
        logits = self.head(h)
        
        # 在分布式环境中，收集所有进程的logits
        world_size, _ = get_distributed_info()
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        
        return logits
    
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_p: float = 0.9):
        """
        使用自回归方式生成文本。
        
        参数:
            prompt_tokens (torch.Tensor): 提示词元序列 [batch_size, prompt_len]
            max_new_tokens (int): 最大生成的新词元数量
            temperature (float): 采样温度，控制随机性
            top_p (float): 核采样的概率阈值
            
        返回:
            torch.Tensor: 生成的完整序列 [batch_size, prompt_len + new_tokens]
        """
        batch_size, prompt_len = prompt_tokens.shape
        device = prompt_tokens.device
        
        # 复制输入序列作为起始点
        tokens = prompt_tokens.clone()
        
        # 第一次前向传播处理整个prompt
        with torch.no_grad():
            self.forward(tokens)
        
        # 逐个生成新token
        for i in range(max_new_tokens):
            # 确保不超过最大序列长度
            if prompt_len + i >= self.max_seq_len:
                break
            
            # 获取下一个token的预测分布
            with torch.no_grad():
                logits = self.forward(tokens[:, -1:], start_pos=prompt_len+i-1)
            
            # 应用温度
            if temperature > 0:
                logits = logits / temperature
                
            # 应用top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for b in range(batch_size):
                    logits[b, sorted_indices[b][sorted_indices_to_remove[b]]] = -float("inf")
            
            # 从分布中采样
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            
            # 添加到生成序列
            tokens = torch.cat([tokens, next_token], dim=1)
            
        return tokens