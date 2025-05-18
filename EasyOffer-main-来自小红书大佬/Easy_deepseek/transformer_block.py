"""
Transformer块模块 - 实现Transformer架构的核心计算单元。
该模块将注意力层和前馈网络组合为完整的Transformer块，构成模型的基本处理单元。
"""

from torch import nn
import torch
from typing import Optional

from config import ModelArgs
from linear_layers import RMSNorm
from attention import MLA
from feedforward import MLP
from moe import MoE


class Block(nn.Module):
    """
    Transformer块实现，组合注意力层和前馈网络。
    
    每个Transformer块包含:
    1. 多头自注意力层，用于捕获序列内的上下文关系
    2. 前馈网络层，可以是标准MLP或混合专家MoE
    3. 两个层归一化，应用于每个子层的输入
    4. 残差连接，确保信息平滑流动
    
    属性:
        attn (nn.Module): 多层注意力机制
        ffn (nn.Module): 前馈网络(MLP或MoE)
        attn_norm (nn.Module): 注意力层的输入归一化
        ffn_norm (nn.Module): 前馈网络的输入归一化
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化Transformer块。
        
        参数:
            layer_id (int): 当前块在模型中的层索引，用于决定使用MLP还是MoE
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        
        # 初始化多层注意力机制
        self.attn = MLA(args)
        
        # 根据层ID决定使用MLP还是MoE作为前馈网络
        # 通常前几层使用标准MLP，后面的层使用MoE以提高模型容量
        if layer_id < args.n_dense_layers:
            self.ffn = MLP(args.dim, args.inter_dim)
        else:
            self.ffn = MoE(args)
        
        # 初始化层归一化
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transformer块的前向传播。
        
        采用Pre-LayerNorm架构，每个子层的输入先经过归一化，输出带有残差连接。
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim]
            start_pos (int): 序列起始位置，用于KV缓存
            freqs_cis (torch.Tensor): 预计算的旋转位置编码
            mask (Optional[torch.Tensor]): 注意力掩码，用于自回归生成
            
        返回:
            torch.Tensor: 处理后的输出张量 [batch_size, seq_len, dim]
        """
        # 注意力子层：归一化 -> 注意力 -> 残差连接
        # 注意：使用Pre-LayerNorm，先归一化再计算
        h = self.attn_norm(x)
        h = self.attn(h, start_pos, freqs_cis, mask)
        x = x + h  # 残差连接
        
        # 前馈网络子层：归一化 -> 前馈网络 -> 残差连接
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h  # 残差连接
        
        return x