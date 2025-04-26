# 数据并行
切分数据，不切分模型。
## torch.nn.DataParallel (DP)
单进程多线程。master gpu将数据切分成n份，然后分发到每个gpu上，同时还将模型副本也分发到每个gpu上，各个gpu独自进行前向计算，计算完后将各自的梯度发回master gpu，master gpu（优化器只在这里）对模型进行梯度更新，然后再将更新完的模型副本同步到各个gpu上。
- 前向过程
1. 在主设备上将输入按 batch 维度切分；
2. 自动将切分后的子 batch 发送到各个 GPU 副本；
3. 每个副本各自执行前向计算。
- 反向过程
1. 各副本计算完梯度后，将梯度汇总（gather）并加总到主模型，再由主模型更新参数。
- 缺陷：
1. 不支持模型并行，因为权重参数更新需要在master gpu上面进行。
2. 权重参数更新需要在master gpu上面进行导致各gpu的显存分配不均匀。
3. 困于 GIL，会带来性能开销，速度很慢。只能单机多卡，单进程多线程，不能混精训练。
（最后那个格子是梯度）
![alt text](image-4.png)
![alt text](image-2.png)
![alt text](image-3.png)


## torch.nn.parallel.DistributedDataParallel (DDP)
多进程。每个gpu有自己的模型副本，切分后的数据分发到各个gpu上，各个gpu独立进行前后向传播计算梯度，并且计算完后用ringallreduce（每个 GPU 都收到其他 GPU 的梯度）进行各gpu的模型的梯度同步，进而独立进行参数更新，每个gpu有自己的优化器。

进程（GPU）之间只传递梯度，这样网络通信就不再是瓶颈。

Ring-ALLreduce就是多个gpu之间直接通信并且同步各个gpu上的模型副本的梯度。
![alt text](image-5.png)


## DeepSpeed
在模型训练的时候，显存占用情况如下：
- 模型参数（FP16）
- 优化器
    - Adam：
        - 梯度参数平滑值
        - 梯度平方参数平滑值
        - 模型参数副本（用于参数更新）
        - 三者都是Float32（不用float16是因为会丢失精度）
- 激活值（前向传播过程中的中间变量，因为在反向传播会用到，所以存了起来）
- 梯度值（FP16）
除了激活值和batchsize以及seq_len有关，其他的都只和模型参数有关
![alt text](image.png)

> 为什么模型训练的时候，优化器需要存两份梯度和一份模型参数，然后优化器外还要存一份模型参数、和梯度？

![alt text](image-7.png)
![alt text](image-1.png)
可以看出来优化器占用是最多的