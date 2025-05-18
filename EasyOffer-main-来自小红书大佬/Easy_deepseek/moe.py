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
    def init(self, args: ModelArgs):
        """初始化门控机制。参数: args (ModelArgs): 模型配置参数"""
        super().init()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))  # 初始化门控权重和偏置
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None  # 特定模型尺寸使用偏置项(这里是针对7B模型)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """门控机制的前向传播。参数: x (torch.Tensor): 输入张量 [batch_size, seq_len, dim] 返回: Tuple[torch.Tensor, torch.Tensor]: - 路由权重：每个选定专家的权重 [batch_size, seq_len, topk] - 专家索引：选择的专家ID [batch_size, seq_len, topk]"""
        scores = Linear(x, self.weight)  # 计算每个输入对每个专家的分数
        scores = scores.softmax(dim=-1, dtype=torch.float32) if self.score_func == "softmax" else scores.sigmoid()  # 根据评分函数转换分数
        original_scores = scores  # 保存原始分数用于后续路由权重
        scores = scores + self.bias if self.bias is not None else scores  # 应用偏置(如果有)
        if self.n_groups > 1:  # 处理分组路由
            scores = scores.view(x.size(0), self.n_groups, -1)  # 将分数重塑为 [batch_size, seq_len, n_groups, experts_per_group]
            group_scores = scores.amax(dim=-1) if self.bias is None else scores.topk(2, dim=-1)[0].sum(dim=-1)  # 计算每组的总分
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]  # 选择得分最高的几个组
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)  # 将未选中的组分数设为负无穷
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]  # 为每个输入选择得分最高的topk个专家
        weights = original_scores.gather(1, indices)  # 获取选中专家的原始分数作为权重
        weights /= weights.sum(dim=-1, keepdim=True) if self.score_func == "sigmoid" else weights  # sigmoid评分需要归一化
        weights *= self.route_scale  # 应用路由缩放因子
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
    def init(self, dim: int, inter_dim: int):
        """初始化专家网络。参数: dim (int): 输入和输出维度 inter_dim (int): 中间隐藏层维度"""
        super().init()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """专家网络的前向传播。参数: x (torch.Tensor): 输入张量 返回: torch.Tensor: 处理后的输出张量"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # 与MLP相同的计算结构

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
    def init(self, args: ModelArgs):
        """初始化混合专家模型。参数: args (ModelArgs): 模型配置参数"""
        super().init()
        world_size, rank = get_distributed_info()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"  # 确保专家数能被进程数整除
        self.n_routed_experts = args.n_routed_experts  # 设置专家相关参数
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts  # 计算当前进程负责的专家范围
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)  # 初始化路由门控
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None for i in range(self.n_routed_experts)])  # 初始化专家列表，只创建当前进程负责的专家
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)  # 初始化共享专家(所有token都会通过)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """混合专家模型的前向传播。参数: x (torch.Tensor): 输入张量 [batch_size, seq_len, dim] 返回: torch.Tensor: 经专家处理后的输出 [batch_size, seq_len, dim]"""
        shape = x.size()  # 保存原始形状并重塑为2D张量以简化处理
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)  # 获取路由权重和专家索引
        y = torch.zeros_like(x)  # 初始化输出张量
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()  # 统计每个专家分配到的样本数
        for i in range(self.experts_start_idx, self.experts_end_idx):  # 处理当前进程负责的专家
            if counts[i] == 0:  # 没有样本路由到此专家，跳过计算
                continue
            expert = self.experts[i]  # 获取当前专家的实例
            idx, top = torch.where(indices == i)  # 找到路由到此专家的样本索引和对应权重索引
            y[idx] += expert(x[idx]) * weights[idx, top, None]  # 计算专家输出并按权重加权
        z = self.shared_experts(x)  # 计算共享专家输出(应用于所有样本)
        world_size, _ = get_distributed_info()
        if world_size > 1:  # 在分布式环境中聚合所有专家的输出
            dist.all_reduce(y)
        return (y + z).view(shape)  # 合并路由专家和共享专家的结果，恢复原始形状
    