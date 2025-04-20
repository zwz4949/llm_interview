> 介绍self-attention 和 cross-attention
Self-Attention 是一种让模型能够关注输入序列中所有位置信息的机制，它计算序列中每个元素与其他所有元素的相关性。完全基于输入序列内部的关系；能够捕捉长距离依赖关系

> 介绍多头注意力

> 介绍MHA, MQA, GQA, 
- 在 MHA（Multi Head Attention） 中，每个头有自己单独的 key-value 对；标准的多头注意力机制，h个Query、Key 和 Value 矩阵。

- 在 MQA（Multi Query Attention） 中只会有一组 key-value 对；多查询注意力的一种变体，也是用于自回归解码的一种注意力机制。与MHA不同的是，MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。

- 在 GQA（Grouped Query Attention）中，会对 attention 进行分组操作，query 被分为 N 组，每个组共享一个 Key 和 Value 矩阵GQA将查询头分成G组，每个组共享一个Key 和 Value 矩阵。GQA-G是指具有G组的grouped-query attention。GQA-1具有单个组，因此具有单个Key 和 Value，等效于MQA。而GQA-H具有与头数相等的组，等效于MHA。

> 位置编码的作用