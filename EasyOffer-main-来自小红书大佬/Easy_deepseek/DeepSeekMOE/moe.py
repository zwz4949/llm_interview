"""
混合专家模型(MoE)模块 - 实现条件计算的专家路由系统。
该模块提供了一种动态路由机制，使模型能选择性地激活不同专家网络处理不同输入。
"""

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from typing import Tuple

from config import ModelArgs
from linear_layers import Linear
from distributed import get_distributed_info
from feedforward import MLP

class Gate(nn.Module):
    """
    混合专家模型中的路由门控机制。
    该门控决定每个输入应路由到哪些专家，支持两种评分函数：
    - softmax：概率分布形式，所有专家权重和为1
    - sigmoid：独立激活方式，经归一化后使用
    还支持专家分组路由，可以先选择组，再选择组内专家，提高效率。
    属性:
        dim (int): 输入特征维度
        topk (int): 每个输入激活的专家数量
        n_groups (int): 专家分组数量
        topk_groups (int): 每个输入激活的组数量
        score_func (str): 评分函数类型("softmax"或"sigmoid")
        route_scale (float): 路由权重的缩放因子
        weight (nn.Parameter): 门控的权重参数
        bias (nn.Parameter): 可选的偏置参数
    """
    def __init__(self, args: ModelArgs):
        """初始化门控机制。参数: args (ModelArgs): 模型配置参数"""
        super().__init__()
        self.dim = args.dim  # 输入特征维度：7168
        self.topk = args.n_activated_experts  # 每个token激活的专家数量：8
        self.n_groups = args.n_expert_groups  # 专家分组数量：8
        self.topk_groups = args.n_limited_groups  # 每个输入激活的组数量：4
        self.score_func = args.score_func  # 评分函数类型："sigmoid"
        self.route_scale = args.route_scale  # 路由权重的缩放因子：2.5
        
        # 门控权重矩阵: [n_routed_experts=256, dim=7168]
        # 总参数量：256 * 7168 = 1,835,008
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        
        # 门控偏置向量: [n_routed_experts=256]
        # 总参数量：256
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        门控机制的前向传播。
        参数: 
            x (torch.Tensor): 输入张量 [batch_size*seq_len, dim=7168]
        返回: 
            Tuple[torch.Tensor, torch.Tensor]: 
            - 路由权重：每个选定专家的权重 [batch_size*seq_len, topk=8]
            - 专家索引：选择的专家ID [batch_size*seq_len, topk=8]
        """
        # 计算每个输入对每个专家的分数
        # scores形状: [batch_size*seq_len, n_routed_experts=256]
        scores = Linear(x, self.weight)
        
        # 根据评分函数转换分数，形状保持不变: [batch_size*seq_len, n_routed_experts=256]
        scores = scores.softmax(dim=-1, dtype=torch.float32) if self.score_func == "softmax" else scores.sigmoid()
        original_scores = scores  # 保存原始分数用于后续路由权重
        
        # 应用偏置(如果有)，形状保持不变: [batch_size*seq_len, n_routed_experts=256]
        scores = scores + self.bias if self.bias is not None else scores
        
        if self.n_groups > 1:  # 处理分组路由
            # 将分数重塑为 [batch_size*seq_len, n_groups=8, experts_per_group=32]
            scores = scores.view(x.size(0), self.n_groups, -1)
            
            # 计算每组的总分，形状: [batch_size*seq_len, n_groups=8]
            group_scores = scores.amax(dim=-1) if self.bias is None else scores.topk(2, dim=-1)[0].sum(dim=-1)
            
            # 选择得分最高的几个组，indices形状: [batch_size*seq_len, topk_groups=4]
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            
            # 生成掩码，形状: [batch_size*seq_len, n_groups=8]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            
            # 将未选中的组分数设为负无穷，形状恢复: [batch_size*seq_len, n_routed_experts=256]
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # 为每个输入选择得分最高的topk个专家，indices形状: [batch_size*seq_len, topk=8]
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        
        # 获取选中专家的原始分数作为权重，weights形状: [batch_size*seq_len, topk=8]
        weights = original_scores.gather(1, indices)
        
        # sigmoid评分需要归一化，形状不变: [batch_size*seq_len, topk=8]
        weights /= weights.sum(dim=-1, keepdim=True) if self.score_func == "sigmoid" else weights
        
        # 应用路由缩放因子，形状不变: [batch_size*seq_len, topk=8]
        weights *= self.route_scale
        
        return weights.type_as(x), indices

class Expert(nn.Module):
    """
    混合专家模型中的单个专家网络。
    结构与MLP类似，但不使用并行线性层，因为专家之间已经并行。
    每个专家独立处理路由到它的部分输入，形成专家系统。
    属性:
        w1 (nn.Module): 输入到中间维度的投影
        w2 (nn.Module): 中间维度到输出的投影
        w3 (nn.Module): 门控投影
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化专家网络。
        参数: 
            dim (int): 输入和输出维度 (7168)
            inter_dim (int): 中间隐藏层维度 (2048)
        """
        super().__init__()
        # 第一投影矩阵: [dim=7168, inter_dim=2048]
        # 参数量: 7168 * 2048 = 14,680,064
        self.w1 = Linear(dim, inter_dim)
        
        # 第二投影矩阵: [inter_dim=2048, dim=7168]
        # 参数量: 2048 * 7168 = 14,680,064
        self.w2 = Linear(inter_dim, dim)
        
        # 门控投影矩阵: [dim=7168, inter_dim=2048]
        # 参数量: 7168 * 2048 = 14,680,064
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        专家网络的前向传播。
        参数: 
            x (torch.Tensor): 输入张量 [token_count, dim=7168]
                             (token_count是路由到此专家的token数)
        返回: 
            torch.Tensor: 处理后的输出张量 [token_count, dim=7168]
        """
        # 与MLP相同的计算结构
        # self.w1(x)形状: [token_count, inter_dim=2048]
        # self.w3(x)形状: [token_count, inter_dim=2048]
        # SiLU激活和乘法后形状: [token_count, inter_dim=2048]
        # 最终输出形状: [token_count, dim=7168]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    """
    混合专家模型(MoE)完整实现。
    MoE层包含多个专家网络和一个路由门控，使模型能够:
    1. 为不同输入动态选择最合适的计算路径
    2. 增加模型参数量同时保持计算效率
    3. 在相同计算量下提高模型容量
    属性:
        dim (int): 输入特征维度
        n_routed_experts (int): 路由专家总数
        n_local_experts (int): 当前进程负责的专家数
        n_activated_experts (int): 每个token激活的专家数
        gate (nn.Module): 路由门控机制
        experts (nn.ModuleList): 专家网络列表
        shared_experts (nn.Module): 所有输入共享的专家网络
    """
    def __init__(self, args: ModelArgs):
        """初始化混合专家模型。参数: args (ModelArgs): 模型配置参数"""
        super().__init__()
        world_size, rank = get_distributed_info()
        self.dim = args.dim  # 输入特征维度: 7168
        
        # 确保专家数能被进程数整除
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        
        # 设置专家相关参数
        self.n_routed_experts = args.n_routed_experts  # 路由专家总数：256
        self.n_local_experts = args.n_routed_experts // world_size  # 当前进程负责的专家数
        self.n_activated_experts = args.n_activated_experts  # 每个token激活的专家数：8
        
        # 计算当前进程负责的专家范围索引
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        
        # 初始化路由门控 - 包含参数约1.8M
        self.gate = Gate(args)
        
        # 初始化专家列表，每个专家大约44M参数
        # 列表长度为n_routed_experts=256，但只创建当前进程负责的专家
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim) 
            if self.experts_start_idx <= i < self.experts_end_idx else None 
            for i in range(self.n_routed_experts)
        ])
        
        # 共享专家，所有token都会通过
        # 参数量与单个Expert相近，约44M参数
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        混合专家模型的前向传播。
        参数: 
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim=7168]
        返回: 
            torch.Tensor: 经专家处理后的输出 [batch_size, seq_len, dim=7168]
        """
        # 保存原始形状并重塑为2D张量以简化处理
        shape = x.size()  # 形状: [batch_size, seq_len, dim=7168]
        x = x.view(-1, self.dim)  # 重塑为: [batch_size*seq_len, dim=7168]
        
        # 获取路由权重和专家索引
        # weights形状: [batch_size*seq_len, topk=8]
        # indices形状: [batch_size*seq_len, topk=8]
        weights, indices = self.gate(x)
        
        # 初始化输出张量: [batch_size*seq_len, dim=7168]
        y = torch.zeros_like(x)
        
        # 统计每个专家分配到的样本数，counts长度: n_routed_experts=256
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        # 处理当前进程负责的专家
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:  # 没有样本路由到此专家，跳过计算
                continue
                
            expert = self.experts[i]  # 获取当前专家的实例
            
            # 找到路由到此专家的样本索引和对应权重索引
            # idx形状: [count_i]，其中count_i是路由到专家i的token数
            # top形状: [count_i]
            idx, top = torch.where(indices == i)
            
            # 计算专家输出并按权重加权
            # expert(x[idx])形状: [count_i, dim=7168]
            # weights[idx, top, None]形状: [count_i, 1]
            # 加权结果添加到y中的对应位置
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        
        # 计算共享专家输出(应用于所有样本)
        # z形状: [batch_size*seq_len, dim=7168]
        z = self.shared_experts(x)
        
        # 在分布式环境中聚合所有专家的输出
        world_size, _ = get_distributed_info()
        if world_size > 1:
            # 聚合后y形状保持不变: [batch_size*seq_len, dim=7168]
            dist.all_reduce(y)
        
        # 合并路由专家和共享专家的结果，恢复原始形状
        # 最终输出: [batch_size, seq_len, dim=7168]
        return (y + z).view(shape)