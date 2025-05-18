"""
线性层模块 - 实现各种类型的线性变换层。
该模块提供标准线性层、行/列并行线性层、嵌入层以及归一化层的实现。
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from config import block_size, gemm_impl
from quantization import act_quant, weight_dequant, fp8_gemm
from distributed import get_distributed_info

def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    量化感知的线性变换函数，支持不同精度下的计算。
    
    根据权重是否量化和全局配置，选择不同的计算路径:
    - 标准线性变换：对于普通浮点权重
    - bf16线性变换：对于已量化但需反量化的权重
    - fp8线性变换：对于保持量化状态的权重
    
    参数:
        x (torch.Tensor): 输入张量
        weight (torch.Tensor): 权重张量，可能已被量化
        bias (torch.Tensor, optional): 偏置项
        
    返回:
        torch.Tensor: 线性变换结果
    """
    if weight.element_size() > 1:
        # 权重未量化，使用标准线性变换
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # 使用bf16实现，先反量化权重
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # 使用fp8实现，保持量化状态进行计算
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    自定义线性层，支持量化权重和可选偏置。
    
    属性:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        weight (nn.Parameter): 权重参数，可能是量化格式
        scale (nn.Parameter): 量化权重的缩放因子(仅当权重量化时存在)
        bias (nn.Parameter): 可选的偏置参数
    """
    dtype = torch.bfloat16  # 默认数据类型，可被设置为fp8

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        """
        初始化线性层。
        
        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            bias (bool): 是否包含偏置项
            dtype: 权重数据类型，默认使用类变量dtype
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        
        # 如果权重是量化格式(element_size为1)，创建缩放因子
        if self.weight.element_size() == 1:
            # 计算量化块的数量
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            # 创建缩放因子参数
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        
        # 初始化偏置(如果需要)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """线性层的前向传播"""
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    列并行线性层，将输出特征分布到不同进程上。
    
    在输出维度上进行模型并行，每个进程只负责部分输出特征的计算，
    从而减小每个GPU上的参数量和计算量。
    
    属性:
        part_out_features (int): 每个进程负责的输出特征数
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        """
        初始化列并行线性层。
        
        参数:
            in_features (int): 输入特征维度
            out_features (int): 总输出特征维度(所有进程合计)
            bias (bool): 是否包含偏置项
            dtype: 权重数据类型
        """
        world_size, _ = get_distributed_info()
        
        # 确保输出特征能被进程数整除
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        
        # 计算每个进程负责的输出特征数
        self.part_out_features = out_features // world_size
        
        # 使用进程特定的输出特征数初始化基类
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """列并行线性层的前向传播"""
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """
    行并行线性层，将输入特征分布到不同进程上。
    
    在输入维度上进行模型并行，每个进程只负责部分输入特征的计算，
    计算后通过all-reduce聚合所有进程的结果。
    
    属性:
        part_in_features (int): 每个进程负责的输入特征数
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        """
        初始化行并行线性层。
        
        参数:
            in_features (int): 总输入特征维度(所有进程合计)
            out_features (int): 输出特征维度
            bias (bool): 是否包含偏置项
            dtype: 权重数据类型
        """
        world_size, _ = get_distributed_info()
        
        # 确保输入特征能被进程数整除
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        
        # 计算每个进程负责的输入特征数
        self.part_in_features = in_features // world_size
        
        # 使用进程特定的输入特征数初始化基类
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行并行线性层的前向传播。
        
        执行当前进程的局部计算，然后通过all-reduce聚合所有进程的结果。
        偏置仅在聚合后添加，避免重复。
        """
        world_size, _ = get_distributed_info()
        
        # 计算本地线性变换(不包括偏置)
        y = linear(x, self.weight)
        
        # 在分布式环境中聚合所有进程的结果
        if world_size > 1:
            dist.all_reduce(y)
        
        # 在聚合后添加偏置(如果有)
        if self.bias is not None:
            y += self.bias
            
        return y


class ParallelEmbedding(nn.Module):
    """
    并行嵌入层，将词汇表分割到不同进程上。
    
    每个进程只存储词汇表的一部分，减少内存占用。
    查找时，仅处理自己负责的词汇部分，并通过all-reduce聚合结果。
    
    属性:
        vocab_size (int): 总词汇表大小
        dim (int): 嵌入维度
        part_vocab_size (int): 每个进程负责的词汇表大小
        vocab_start_idx (int): 当前进程负责的词汇表起始索引
        vocab_end_idx (int): 当前进程负责的词汇表结束索引
        weight (nn.Parameter): 嵌入权重参数
    """
    def __init__(self, vocab_size: int, dim: int):
        """
        初始化并行嵌入层。
        
        参数:
            vocab_size (int): 总词汇表大小
            dim (int): 嵌入维度
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        world_size, rank = get_distributed_info()
        
        # 确保词汇表大小能被进程数整除
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        
        # 计算当前进程负责的词汇表部分
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        
        # 只存储当前进程负责的词汇部分的嵌入
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        并行嵌入层的前向传播。
        
        处理当前进程负责的词汇部分，其他部分置零，
        然后通过all-reduce聚合所有进程的结果。
        
        参数:
            x (torch.Tensor): 包含词索引的输入张量
            
        返回:
            torch.Tensor: 嵌入后的张量
        """
        world_size, _ = get_distributed_info()
        
        if world_size > 1:
            # 创建掩码，标记不属于当前进程词汇范围的token
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            
            # 调整索引以匹配当前进程的词汇部分
            x = x - self.vocab_start_idx
            
            # 将不属于当前进程的词索引设为0，防止越界
            x[mask] = 0
        
        # 执行嵌入查找
        y = F.embedding(x, self.weight)
        
        if world_size > 1:
            # 将不属于当前进程的嵌入置零
            y[mask] = 0
            
            # 聚合所有进程的嵌入结果
            dist.all_reduce(y)
            
        return y


class RMSNorm(nn.Module):
    """
    均方根层归一化(Root Mean Square Layer Normalization)。
    
    相比LayerNorm，RMSNorm移除了均值中心化步骤，只对均方根进行归一化，
    计算效率更高，且在Transformer模型中表现良好。
    
    属性:
        dim (int): 要归一化的特征维度
        eps (float): 数值稳定性的小常数
        weight (nn.Parameter): 可学习的缩放参数
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化RMSNorm层。
        
        参数:
            dim (int): 要归一化的特征维度
            eps (float): 数值稳定性的小常数
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # 初始化缩放参数为1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        RMSNorm的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 归一化后的张量
        """
        # 使用PyTorch的RMS归一化函数
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)