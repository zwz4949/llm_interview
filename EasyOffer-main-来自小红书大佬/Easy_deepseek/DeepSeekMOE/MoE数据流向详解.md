# MoE数据流向详解：以DeepSeek-671B为例

## 目录
- [1. DeepSeek-671B配置参数](#1-deepseek-671b配置参数)
- [2. 参数规模总结](#2-参数规模总结每个模块)
- [3. MoE门控路由详细示例](#3-moe门控路由详细示例一个更加简单的例子)
- [4. Token在MoE中的独立性与结果拼接机制](#4-token在moe中的独立性与结果拼接详解)

---

## 1. DeepSeek-671B配置参数

```json
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "fp8"
}
```

## 2. 参数规模总结（每个模块）

| 模块 | 参数张量大小 |
|------|-------------|
| Embedding | $129280 \times 7168$ |
| Attention QKV | $3 \times (7168 \times 16384)$ |
| Attention O | $16384 \times 7168$ |
| LoRA Q | $7168 \times 1536$, $1536 \times 16384$ |
| LoRA KV | $7168 \times 512$, $512 \times 16384$ |
| Dense MLP (每层) | $7168 \times 18432$, $18432 \times 7168$, $7168 \times 18432$ |
| MoE Gate (每MoE层) | $256 \times 7168$ |
| MoE Experts (每个专家) | $7168 \times 2048$, $2048 \times 7168$, $7168 \times 2048$ |
| MoE Shared Expert | $7168 \times 2048$, $2048 \times 7168$, $7168 \times 2048$ |
| LayerNorm (每层) | $7168 \times 2$ |
| LM Head | $7168 \times 129280$ |

---

## 3. MoE门控路由详细示例（一个更加简单的例子）

基于配置参数：`dim=16, n_experts=32, n_groups=8, topk_groups=2, topk=2`，详细追踪批次输入张量`[2, 4, 16]`（2个序列，每个序列4个token，每个token 16维特征）通过MoE模型的完整计算过程。

### 3.1 输入数据准备

为了具体展示，我们创建一个样本输入：

```python
x = torch.tensor([
    # Batch 1 (4个token)
    [[0.5, 0.2, 0.3, -0.1, 0.7, -0.2, 0.4, 0.1, -0.3, 0.6, -0.4, 0.8, 0.2, -0.5, 0.3, 0.9],  # token 1
     [0.3, -0.2, 0.8, 0.4, -0.5, 0.6, 0.1, -0.7, 0.2, 0.9, -0.3, 0.5, 0.0, -0.4, 0.7, 0.2],  # token 2
     [-0.4, 0.7, -0.1, 0.6, -0.8, 0.3, -0.5, 0.2, -0.6, 0.1, -0.9, 0.4, -0.2, 0.8, -0.3, 0.5], # token 3
     [0.2, 0.6, -0.3, 0.8, -0.1, 0.7, -0.4, 0.9, -0.2, 0.5, -0.6, 0.1, -0.7, 0.3, -0.8, 0.4]], # token 4
    
    # Batch 2 (4个token)
    [[0.6, -0.3, 0.8, -0.2, 0.9, -0.5, 0.7, -0.1, 0.4, -0.6, 0.2, -0.8, 0.3, -0.7, 0.1, -0.4], # token 5
     [-0.5, 0.2, -0.9, 0.3, -0.7, 0.4, -0.1, 0.8, -0.6, 0.5, -0.3, 0.7, -0.2, 0.6, -0.4, 0.1], # token 6
     [0.8, -0.4, 0.1, -0.9, 0.5, -0.3, 0.7, -0.2, 0.6, -0.1, 0.3, -0.5, 0.2, -0.8, 0.4, -0.6], # token 7
     [-0.7, 0.3, -0.5, 0.2, -0.8, 0.6, -0.4, 0.1, -0.3, 0.9, -0.2, 0.7, -0.1, 0.4, -0.6, 0.5]]  # token 8
])

# 输入张量形状: [2, 4, 16]
# - 2个批次
# - 每批次4个token
# - 每个token 16维特征
```

### 3.2 Gate类处理流程详解

#### 3.2.1 输入准备

```python
# MoE.forward中首先保存原始形状并重塑输入
shape = x.size()  # 保存形状: [2, 4, 16]
x = x.view(-1, self.dim)  # 重塑为: [8, 16]

# x现在包含8个token的特征向量
# token 1-4来自批次1，token 5-8来自批次2
```

#### 3.2.2 Gate初始化参数

```python
# Gate实例参数
self.dim = 16                 # 输入特征维度
self.topk = 2                 # 每个token激活的专家数
self.n_groups = 8             # 专家分组数量
self.topk_groups = 2          # 每个token激活的组数量
self.score_func = "sigmoid"   # 评分函数类型
self.route_scale = 2.5        # 路由权重缩放因子

# 权重矩阵示例（部分展示）
self.weight = nn.Parameter(torch.tensor([
    [ 0.03, -0.02,  0.01, ...,  0.02], # 专家1的权重
    [-0.01,  0.04,  0.02, ..., -0.03], # 专家2的权重
    ...
    [ 0.02, -0.01,  0.03, ...,  0.01]  # 专家32的权重
]))  # 形状: [32, 16]
```

#### 3.2.3 计算专家亲和度分数

```python
# 执行 scores = Linear(x, self.weight)
scores = x @ self.weight.T  # [8, 16] @ [16, 32] -> [8, 32]

# 具体计算结果（显示前5列）
scores = tensor([
    [ 0.21,  0.15,  0.18,  0.27, -0.12, ...], # token 1的分数
    [ 0.19,  0.23,  0.14,  0.31,  0.26, ...], # token 2的分数
    [-0.16,  0.22, -0.13,  0.19, -0.24, ...], # token 3的分数
    [ 0.28,  0.17,  0.32,  0.13,  0.29, ...], # token 4的分数
    [ 0.35, -0.14,  0.27, -0.18,  0.33, ...], # token 5的分数
    [-0.22,  0.16, -0.29,  0.21, -0.17, ...], # token 6的分数
    [ 0.31, -0.25,  0.18, -0.13,  0.24, ...], # token 7的分数
    [-0.19,  0.28, -0.15,  0.23, -0.21, ...], # token 8的分数
])
```

#### 3.2.4 转换为概率值

```python
# 应用sigmoid激活函数
scores = scores.sigmoid()
original_scores = scores.clone()  # 保存原始分数副本

scores = tensor([
    [ 0.55,  0.54,  0.54,  0.57,  0.47, ...], # token 1 
    [ 0.55,  0.56,  0.53,  0.58,  0.56, ...], # token 2
    [ 0.46,  0.55,  0.47,  0.55,  0.44, ...], # token 3
    [ 0.57,  0.54,  0.58,  0.53,  0.57, ...], # token 4
    [ 0.59,  0.47,  0.57,  0.46,  0.58, ...], # token 5
    [ 0.45,  0.54,  0.43,  0.55,  0.46, ...], # token 6
    [ 0.58,  0.44,  0.54,  0.47,  0.56, ...], # token 7
    [ 0.45,  0.57,  0.46,  0.56,  0.45, ...], # token 8
])
```

#### 3.2.5 分组重塑
```python
# 重塑为组结构 - 每组4个专家
scores = scores.view(8, 8, 4)  # [8, n_groups, experts_per_group]

# 第一个token的分组分数示例:
token1_scores = tensor([
    [0.55, 0.54, 0.54, 0.57],  # 组1: 专家1-4的分数
    [0.47, 0.58, 0.56, 0.55],  # 组2: 专家5-8的分数
    [0.56, 0.57, 0.58, 0.54],  # 组3: 专家9-12的分数
    [0.48, 0.55, 0.57, 0.55],  # 组4: 专家13-16的分数
    [0.54, 0.56, 0.58, 0.53],  # 组5: 专家17-20的分数
    [0.56, 0.58, 0.54, 0.55],  # 组6: 专家21-24的分数
    [0.57, 0.53, 0.55, 0.55],  # 组7: 专家25-28的分数
    [0.57, 0.54, 0.59, 0.56]   # 组8: 专家29-32的分数
])
```

#### 3.2.6 计算组得分

```python
# 获取每组最大值作为组得分
group_scores = scores.amax(dim=-1)  # [8, 8]

group_scores = tensor([
    [ 0.57, 0.58, 0.58, 0.57, 0.58, 0.58, 0.57, 0.59], # token 1
    [ 0.56, 0.59, 0.58, 0.57, 0.57, 0.59, 0.58, 0.57], # token 2
    [ 0.55, 0.56, 0.58, 0.57, 0.54, 0.56, 0.55, 0.58], # token 3
    [ 0.58, 0.57, 0.59, 0.58, 0.58, 0.57, 0.56, 0.59], # token 4
    [ 0.59, 0.58, 0.57, 0.58, 0.59, 0.58, 0.57, 0.58], # token 5
    [ 0.54, 0.57, 0.58, 0.57, 0.55, 0.56, 0.57, 0.59], # token 6
    [ 0.58, 0.57, 0.58, 0.59, 0.57, 0.58, 0.57, 0.58], # token 7
    [ 0.57, 0.56, 0.57, 0.58, 0.58, 0.57, 0.59, 0.58]  # token 8
])
```

#### 3.2.7 选择最佳组

```python
# 为每个token选择得分最高的两个组
_, indices = group_scores.topk(2, dim=-1)

indices = tensor([
    [7, 2], # token 1: 选中组8和组3
    [5, 1], # token 2: 选中组6和组2
    [7, 2], # token 3: 选中组8和组3
    [7, 2], # token 4: 选中组8和组3
    [0, 4], # token 5: 选中组1和组5
    [7, 2], # token 6: 选中组8和组3
    [3, 0], # token 7: 选中组4和组1
    [6, 3]  # token 8: 选中组7和组4
])
```

#### 3.2.8 创建组掩码

```python
# 创建初始全1掩码
mask = torch.ones(8, 8, dtype=bool)  # [8, 8]

# 将选中组的位置设置为False
for i in range(8):  # 对每个token
    mask[i].scatter_(0, indices[i], False)

# 生成的掩码矩阵
mask = tensor([
    [ True,  True, False,  True,  True,  True,  True, False], # token 1
    [ True, False,  True,  True,  True, False,  True,  True], # token 2
    [ True,  True, False,  True,  True,  True,  True, False], # token 3
    [ True,  True, False,  True,  True,  True,  True, False], # token 4
    [False,  True,  True,  True, False,  True,  True,  True], # token 5
    [ True,  True, False,  True,  True,  True,  True, False], # token 6
    [False,  True,  True, False,  True,  True,  True,  True], # token 7
    [ True,  True,  True, False,  True,  True, False,  True]  # token 8
])
```

#### 3.2.9 应用组掩码

```python
# 将未选中组的专家得分设为负无穷
scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf"))

# token 1的掩码后分数示例:
token1_masked_scores = tensor([
    [-inf, -inf, -inf, -inf],  # 组1 (未选中)
    [-inf, -inf, -inf, -inf],  # 组2 (未选中)
    [0.56, 0.57, 0.58, 0.54],  # 组3 (选中)
    [-inf, -inf, -inf, -inf],  # 组4 (未选中)
    [-inf, -inf, -inf, -inf],  # 组5 (未选中)
    [-inf, -inf, -inf, -inf],  # 组6 (未选中)
    [-inf, -inf, -inf, -inf],  # 组7 (未选中)
    [0.57, 0.54, 0.59, 0.56]   # 组8 (选中)
])
```

#### 3.2.10 展平回专家视图

```python
# 恢复专家视图
scores = scores.flatten(1)  # [8, 32]

# token 1展平后的前16个专家分数:
token1_flat_scores = tensor([
    -inf, -inf, -inf, -inf,    # 组1的4个专家 (未选中)
    -inf, -inf, -inf, -inf,    # 组2的4个专家 (未选中)
    0.56, 0.57, 0.58, 0.54,    # 组3的4个专家 (选中)
    -inf, -inf, -inf, -inf,    # 组4的4个专家 (未选中)
    ...                        # 后续组
])
```

#### 3.2.11 选择顶部专家

```python
# 选择得分最高的2个专家
_, indices = torch.topk(scores, 2, dim=-1)

indices = tensor([
    [30, 10], # token 1: 选择专家31和专家11
    [21,  6], # token 2: 选择专家22和专家7
    [31, 11], # token 3: 选择专家32和专家12
    [30,  9], # token 4: 选择专家31和专家10
    [ 0, 16], # token 5: 选择专家1和专家17
    [31, 10], # token 6: 选择专家32和专家11
    [15,  0], # token 7: 选择专家16和专家1
    [26, 15]  # token 8: 选择专家27和专家16
])
```

#### 3.2.12 获取专家权重

```python
# 从原始分数中提取选中专家的分数
weights = original_scores.gather(1, indices)

weights = tensor([
    [0.59, 0.58], # token 1的两个选中专家的分数
    [0.59, 0.58], # token 2的两个选中专家的分数
    [0.58, 0.58], # token 3的两个选中专家的分数
    [0.59, 0.57], # token 4的两个选中专家的分数
    [0.59, 0.59], # token 5的两个选中专家的分数
    [0.59, 0.58], # token 6的两个选中专家的分数
    [0.59, 0.58], # token 7的两个选中专家的分数
    [0.59, 0.58]  # token 8的两个选中专家的分数
])
```

#### 3.2.13 权重归一化与缩放

```python
# 对sigmoid结果进行归一化
weights /= weights.sum(dim=-1, keepdim=True)

weights = tensor([
    [0.504, 0.496], # token 1归一化权重
    [0.504, 0.496], # token 2归一化权重
    [0.500, 0.500], # token 3归一化权重
    [0.509, 0.491], # token 4归一化权重
    [0.500, 0.500], # token 5归一化权重
    [0.504, 0.496], # token 6归一化权重
    [0.504, 0.496], # token 7归一化权重
    [0.504, 0.496]  # token 8归一化权重
])

# 应用路由缩放
weights *= self.route_scale  # 缩放因子为2.5

weights = tensor([
    [1.261, 1.239], # token 1最终权重
    [1.261, 1.239], # token 2最终权重
    [1.250, 1.250], # token 3最终权重
    [1.271, 1.229], # token 4最终权重
    [1.250, 1.250], # token 5最终权重
    [1.261, 1.239], # token 6最终权重
    [1.261, 1.239], # token 7最终权重
    [1.261, 1.239]  # token 8最终权重
])
```

### 3.3 MoE类处理流程详解

#### 3.3.1 门控返回值

```python
# 经过Gate处理，返回值为:
# - weights: [8, 2] (每个token的2个专家权重)
# - indices: [8, 2] (每个token选择的2个专家索引)
weights, indices = self.gate(x)
```

#### 3.3.2 初始化输出

```python
# 创建空输出张量
y = torch.zeros_like(x)  # [8, 16] 全零矩阵
```

#### 3.3.3 统计专家负载

```python
# 将indices展平并计数每个专家被选中的次数
counts = torch.bincount(indices.flatten(), minlength=32)

counts = tensor([
    2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 3, 1, 0, 0, 0, 2,   # 专家1-16
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2    # 专家17-32
])

# 例如，专家1被选中2次，专家31被选中2次，专家11被选中3次
```

#### 3.3.4 处理专家31（示例）

```python
# 处理专家31 (索引30)
i = 30
idx, top = torch.where(indices == i)
# idx = tensor([0, 3])  # token 1和token 4选择了专家31
# top = tensor([0, 0])  # 两者都作为第一选择

expert = self.experts[30]  # 获取专家31的实例

# 提取要处理的token特征
token_features = x[idx]  # 获取token 1和token 4的特征
# token_features形状: [2, 16]

# 执行专家计算
expert_out = expert(token_features)
# 假设输出:
expert_out = tensor([
    [0.42, -0.35, 0.61, 0.24, -0.53, 0.75, 0.38, -0.47, 0.83, 0.51, -0.29, 0.92, 0.45, -0.67, 0.79, 0.33],  # token 1输出
    [0.39, 0.28, -0.41, 0.73, -0.22, 0.64, -0.36, 0.81, -0.25, 0.57, -0.59, 0.20, -0.63, 0.44, -0.71, 0.52]   # token 4输出
])

# 提取对应权重并扩展维度用于广播
weight_values = weights[idx, top].unsqueeze(-1)  # 形状: [2, 1]
# weight_values = tensor([[1.261], [1.271]])

# 应用权重
weighted_out = expert_out * weight_values
# weighted_out = tensor([
#     [0.530, -0.441, 0.769, 0.303, -0.668, 0.946, 0.479, -0.593, 1.047, 0.643, -0.366, 1.160, 0.567, -0.845, 0.996, 0.416],
#     [0.496, 0.356, -0.521, 0.928, -0.280, 0.814, -0.458, 1.030, -0.318, 0.725, -0.750, 0.254, -0.801, 0.559, -0.903, 0.661]
# ])

# 将权重结果添加到输出张量
y[idx] += weighted_out
```

#### 3.3.5 处理专家11（示例）

```python
# 处理专家11 (索引10)
i = 10
idx, top = torch.where(indices == i)
# idx = tensor([0, 5, 6])  # token 1, token 6和token 7选择了专家11
# top = tensor([1, 1, 1])  # 都作为第二选择

expert = self.experts[10]  # 获取专家11的实例

# 提取要处理的token特征和计算
token_features = x[idx]  # 形状: [3, 16]
expert_out = expert(token_features)  # 形状: [3, 16]

# 提取对应权重并应用
weight_values = weights[idx, top].unsqueeze(-1)  # 形状: [3, 1]
weighted_out = expert_out * weight_values

# 更新输出
y[idx] += weighted_out
```

#### 3.3.6 处理所有其他专家

```python
# 类似地处理所有被至少一个token选择的专家
# 每个专家只处理路由到它的token，大大减少计算量
```

#### 3.3.7 共享专家计算

```python
# 所有token通过共享专家
z = self.shared_experts(x)  # 形状: [8, 16]

# 共享专家是一个普通MLP，处理所有8个token
```

#### 3.3.8 合并结果

```python
# 合并路由专家和共享专家的结果
output = y + z  # 形状: [8, 16]

# 恢复原始批次形状
output = output.view(shape)  # 形状: [2, 4, 16]
```

#### 3.3.9 最终输出

MoE模块输出形状为`[2, 4, 16]`的张量，与输入形状相同，但内容经过了专家网络的处理。

### 3.4 过程可视化

路由过程可以概括为：
1. 每个token经过两阶段路由：
   - 先选择最相关的2个专家组（共8个专家）
   - 再从这8个专家中选择最相关的2个专家
2. 每个token最终只激活2个路由专家和1个共享专家
3. 专家输出按权重加权后合并

例如，token 1的路由路径：
```
原始输入 → 选择组3和组8 → 选择专家11和专家31 → 专家处理并加权 → 合并共享专家结果 → 最终输出
```

---

## 4. Token在MoE中的独立性与结果拼接详解

### 4.1 Token的含义与作用

#### 4.1.1 Token基本概念

在自然语言处理中，"token"是文本的基本单位：

- **定义**：一个token可以是一个单词、子词(subword)、字符或特殊标记
- **示例**：句子"I love AI"可被分词为["I", "love", "AI"]三个token
- **表示方式**：每个token在模型中被表示为一个向量（我们例子中是16维向量）

#### 4.1.2 在我们的例子中

在`[2, 4, 16]`的输入中：
- 2个批次（可能是2个不同的文本样本）
- 每个批次有4个token（可能是4个单词或子词）
- 每个token用16维向量表示其特征

### 4.2 Token之间的独立性与交互

#### 4.2.1 MoE层中的独立性

**在MoE层内，token处理是相互独立的**：

1. **路由独立性**：每个token根据自身特征独立选择专家
2. **计算独立性**：专家处理每个token时不考虑其他token
3. **结果独立性**：每个token的输出只依赖于它选择的专家

这种独立处理非常高效，允许并行计算，也是MoE能处理大量参数的关键。

#### 4.2.2 整体模型中的交互

虽然在MoE层内token是独立的，但在整个模型中token会产生交互：

1. **通过注意力机制**：在Transformer架构中，自注意力层允许token互相"观察"
   ```
   MoE之前：注意力层 → token间交互
   MoE层内：token独立处理
   MoE之后：注意力层 → token再次交互
   ```

2. **信息流动**：随着层数加深，每个token的表示逐渐包含序列上下文信息

### 4.3 结果拼接的详细机制

这是一个精妙的实现细节，涉及到如何确保正确的结果映射。

#### 4.3.1 基于索引的直接更新

MoE使用索引机制确保输出位置正确：

```python
# 关键代码解析
y = torch.zeros_like(x)  # 初始化输出张量，与输入形状相同 [8, 16]

# 对于每个专家i:
for i in range(self.n_routed_experts):
    if counts[i] == 0:  # 跳过没有token路由到的专家
        continue
    
    idx, top = torch.where(indices == i)
    expert_out = self.experts[i](x[idx])
    y[idx] += expert_out * weights[idx, top, None]
```

#### 4.3.2 详细步骤分析

让我们用具体示例解析这个过程，以专家31（索引30）为例：

1. **索引定位**：
   ```python
   idx, top = torch.where(indices == 30)
   # idx = tensor([0, 3])  # token 1和token 4选择了专家31
   # top = tensor([0, 0])  # 两者都将专家31作为第一选择
   ```

2. **特征获取**：
   ```python
   token_features = x[idx]  # 从x中提取索引0和3的特征
   # 形状: [2, 16]
   # 这里提取了两个16维向量，对应token 1和token 4
   ```

3. **专家计算**：
   ```python
   expert_out = expert(token_features)  # 专家31处理这两个token
   # 形状: [2, 16]
   # 输出也是两个16维向量
   ```

4. **权重获取**：
   ```python
   weight_values = weights[idx, top].unsqueeze(-1)
   # idx=[0,3], top=[0,0]意味着取weights的[0,0]和[3,0]位置
   # weights[0,0]=1.261, weights[3,0]=1.271
   # 扩展维度后形状: [2, 1]
   ```

5. **加权处理**：
   ```python
   weighted_out = expert_out * weight_values
   # 形状: [2, 16]
   # 对专家输出按对应权重缩放
   ```

6. **结果更新**：
   ```python
   y[idx] += weighted_out
   # 将结果加到y中对应位置
   # y[0] += weighted_out[0]  # 更新token 1的结果
   # y[3] += weighted_out[1]  # 更新token 4的结果
   ```

#### 4.3.3 关键保障机制

**为什么这种方法能确保正确拼接？**

1. **索引一致性**：
   - `idx`同时用于从输入提取和向输出放回
   - 保证了每个token的输入和输出位置一致性

2. **累加机制**：
   - `+=`操作允许来自不同专家的结果累加
   - 每个token最终结果 = 专家1结果×权重1 + 专家2结果×权重2

3. **完全映射**：
   - 每个token被路由到的所有专家都会贡献到其最终结果
   - 不会有专家输出丢失或错位的问题

### 4.4 具体追踪示例

让我们完整追踪一个token的处理过程：

#### Token 1的完整路径

1. **初始状态**：
   ```
   x[0] = [0.5, 0.2, 0.3, ..., 0.9]  # 16维向量
   y[0] = [0.0, 0.0, 0.0, ..., 0.0]  # 全零初始化
   ```

2. **专家31处理**：
   ```
   # 专家31是token 1的第一选择(权重1.261)
   y[0] += expert31(x[0]) * 1.261
   # y[0]现在包含专家31的加权结果
   ```

3. **专家11处理**：
   ```
   # 专家11是token 1的第二选择(权重1.239)
   y[0] += expert11(x[0]) * 1.239
   # y[0]现在累加了专家11的加权结果
   ```

4. **共享专家处理**：
   ```
   # 所有token都通过共享专家
   z[0] = shared_expert(x[0])
   ```

5. **最终结果**：
   ```
   output[0] = y[0] + z[0]
   # 合并所有专家的贡献
   ```

### 4.5 多线程和并行处理注意事项

在实际实现中，专家计算可能是并行的，这会带来一些额外考虑：

1. **原子操作**：当多个专家并行处理并更新同一token位置时，需要确保原子加操作

2. **顺序无关性**：由于使用加法，专家计算和更新的顺序不影响最终结果：
   ```python
   y[idx] += expertA(x[idx]) * weightA
   y[idx] += expertB(x[idx]) * weightB
   ```
   与下面顺序等价：
   ```python
   y[idx] += expertB(x[idx]) * weightB
   y[idx] += expertA(x[idx]) * weightA
   ```

3. **负载均衡**：MoE实现通常包含负载均衡机制，防止某些专家过载
