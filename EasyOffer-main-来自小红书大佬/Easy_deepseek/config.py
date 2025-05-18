"""
配置模块 - 存储Transformer模型的所有配置参数和超参数。
该模块为整个模型架构提供中心化的配置管理。
"""

from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelArgs:
    """
    模型参数配置类，包含所有模型超参数和结构定义。
    """
    # 基本参数
    max_batch_size: int = 8                # 最大批处理大小，影响内存占用和KV缓存大小
    max_seq_len: int = 4096 * 4            # 最大序列长度，支持超长上下文
    dtype: Literal["bf16", "fp8"] = "bf16" # 计算的数据类型，影响精度和速度
    vocab_size: int = 102400               # 词汇表大小，影响嵌入层和输出层维度
    dim: int = 2048                        # 模型隐藏维度，即Transformer的基本表示维度
    inter_dim: int = 10944                 # MLP中间层维度，影响模型容量
    moe_inter_dim: int = 1408              # MoE专家中间层维度
    n_layers: int = 27                     # Transformer层数
    n_dense_layers: int = 1                # 使用密集MLP的层数(其余使用MoE)
    n_heads: int = 16                      # 注意力头数
    
    # 混合专家模型(MoE)相关参数
    n_routed_experts: int = 64             # 路由专家总数
    n_shared_experts: int = 2              # 共享专家数量(不进行路由)
    n_activated_experts: int = 6           # 每个输入激活的专家数量
    n_expert_groups: int = 1               # 专家组数
    n_limited_groups: int = 1              # 路由限制的组数
    score_func: Literal["softmax", "sigmoid"] = "softmax"  # 路由评分函数
    route_scale: float = 1.                # 路由评分的缩放因子
    
    # 多层注意力(MLA)相关参数
    q_lora_rank: int = 0                   # 查询投影的低秩适配参数
    kv_lora_rank: int = 512                # 键值投影的低秩适配参数
    qk_nope_head_dim: int = 128            # 无位置编码的查询-键头维度
    qk_rope_head_dim: int = 64             # 有旋转位置编码的查询-键头维度
    v_head_dim: int = 128                  # 值头维度
    
    # 旋转位置编码(RoPE)和序列扩展相关参数
    original_seq_len: int = 4096           # 预训练的原始序列长度
    rope_theta: float = 10000.0            # RoPE基础频率
    rope_factor: float = 40                # 序列扩展因子
    beta_fast: int = 32                    # 快速beta修正因子(用于NTK-aware插值)
    beta_slow: int = 1                     # 慢速beta修正因子
    mscale: float = 1.                     # 序列扩展的注意力缩放因子

# 全局配置参数
block_size: int = 128  # 量化操作的块大小
gemm_impl: Literal["bf16", "fp8"] = "bf16"  # 矩阵乘法的实现类型
attn_impl: Literal["naive", "absorb"] = "absorb"  # 注意力计算的实现方式