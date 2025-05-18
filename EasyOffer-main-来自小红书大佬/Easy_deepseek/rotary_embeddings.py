"""
旋转位置编码模块 - 实现RoPE(Rotary Position Embedding)技术。
该模块提供位置信息的编码方法，使模型能感知序列中token的相对位置关系。
"""

import math
import torch
from config import ModelArgs

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算旋转位置编码的复数指数值。
    
    实现了NTK-aware RoPE，支持序列长度扩展，使模型能处理比预训练序列更长的输入。
    
    参数:
        args (ModelArgs): 包含位置编码相关参数的模型配置
        
    返回:
        torch.Tensor: 预计算的复数指数值，用于高效实现旋转操作
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """计算旋转位置编码的校正维度"""
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """计算校正维度范围，用于NTK-aware RoPE插值"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min_, max_, dim):
        """生成线性插值因子，用于平滑过渡"""
        if min_ == max_:
            max_ += 0.001  # 避免除零错误
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 计算基本频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # 如果序列长度超过预训练长度，应用NTK-aware插值
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 应用插值，使频率适应更长序列
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 为所有位置生成频率
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    
    # 转换为复数指数形式，便于后续旋转操作
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码应用于输入张量。
    
    通过复数乘法实现旋转操作，高效地将位置信息编码到特征向量中。
    
    参数:
        x (torch.Tensor): 输入张量，通常是查询或键向量
        freqs_cis (torch.Tensor): 预计算的复数指数值
        
    返回:
        torch.Tensor: 应用旋转位置编码后的张量
    """
    # 保存原始数据类型
    dtype = x.dtype
    
    # 将张量重新解释为复数形式
    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    
    # 调整freqs_cis形状以便广播
    freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    
    # 使用复数乘法执行旋转
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    
    # 转回原始数据类型
    return x_rotated.to(dtype)