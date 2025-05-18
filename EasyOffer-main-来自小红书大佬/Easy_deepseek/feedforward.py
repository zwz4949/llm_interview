"""
前馈网络模块 - 实现Transformer块中的MLP层。
该模块提供具有门控机制的前馈网络实现，在模型中用于非线性特征转换。
"""

import torch
import torch.nn.functional as F
from torch import nn

from linear_layers import ColumnParallelLinear, RowParallelLinear

class MLP(nn.Module):
    """
    多层感知器(MLP)，用作Transformer块中的前馈网络层。
    
    该实现使用SiLU激活函数和乘法门控机制，在LLM中表现优异。
    门控机制使网络能够动态调整信息流，提高模型表达能力。
    
    结构: FC1和FC3并行处理输入，FC2整合结果
    - FC1: 输入投影到中间维度，后接激活函数
    - FC3: 输入到门控投影
    - FC2: 激活后的FC1输出与FC3输出相乘，再投影回原始维度
    
    属性:
        w1 (nn.Module): 输入到中间维度的投影
        w2 (nn.Module): 中间维度到输出维度的投影
        w3 (nn.Module): 门控投影
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化MLP层。
        
        参数:
            dim (int): 输入和输出维度
            inter_dim (int): 中间隐藏层维度(通常是dim的4倍左右)
        """
        super().__init__()
        
        # 输入到中间维度的投影(使用列并行减少每个GPU的负担)
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        
        # 中间维度到输出的投影(使用行并行聚合结果)
        self.w2 = RowParallelLinear(inter_dim, dim)
        
        # 门控机制的投影
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP的前向传播。
        
        实现公式: output = w2(silu(w1(x)) * w3(x))
        其中silu是激活函数，*表示逐元素乘法(门控机制)。
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim]
            
        返回:
            torch.Tensor: 输出张量 [batch_size, seq_len, dim]
        """
        # 并行计算w1和w3投影
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        
        # 应用SiLU激活函数到w1输出，并与w3输出相乘(门控)
        gated_output = F.silu(w1_out) * w3_out
        
        # 投影回原始维度
        return self.w2(gated_output)