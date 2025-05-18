"""
主模块 - 组装所有组件并提供使用示例。
该模块展示如何初始化和使用Transformer模型进行推理和生成。
"""

import os
import time
import argparse
import torch

from config import ModelArgs
from transformer import Transformer
from distributed import initialize_distributed, get_distributed_info

def setup_model_args(
    model_size: str = "7b", 
    max_seq_len: int = 4096,
    dtype: str = "bf16",
    use_moe: bool = True
) -> ModelArgs:
    """
    根据指定的模型大小和配置设置模型参数。
    
    参数:
        model_size (str): 模型大小标识，如'7b', '13b', '70b'等
        max_seq_len (int): 最大序列长度
        dtype (str): 计算精度，'bf16'或'fp8'
        use_moe (bool): 是否使用混合专家模型
        
    返回:
        ModelArgs: 模型配置参数
    """
    args = ModelArgs()
    
    # 设置最大序列长度和数据类型
    args.max_seq_len = max_seq_len
    args.dtype = dtype
    
    # 根据模型大小设置参数
    if model_size == "7b":
        args.dim = 4096
        args.n_heads = 32
        args.n_layers = 32
        args.vocab_size = 32000
        args.inter_dim = 11008
        if use_moe:
            args.moe_inter_dim = 4096
            args.n_routed_experts = 8
    elif model_size == "13b":
        args.dim = 5120
        args.n_heads = 40
        args.n_layers = 40
        args.vocab_size = 32000
        args.inter_dim = 13696
        if use_moe:
            args.moe_inter_dim = 5120
            args.n_routed_experts = 8
    elif model_size == "70b":
        args.dim = 8192
        args.n_heads = 64
        args.n_layers = 80
        args.vocab_size = 32000
        args.inter_dim = 28672
        if use_moe:
            args.moe_inter_dim = 8192
            args.n_routed_experts = 8
    else:
        # 使用默认值
        pass
        
    # 如果不使用MoE，使所有层为密集层
    if not use_moe:
        args.n_dense_layers = args.n_layers
        
    return args

def load_model(model: Transformer, checkpoint_path: str):
    """
    从检查点加载模型权重。
    
    参数:
        model (Transformer): 模型实例
        checkpoint_path (str): 检查点路径
    """
    world_size, rank = get_distributed_info()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # 根据分布式环境加载检查点
    if world_size > 1:
        # 分片加载，每个进程仅加载自己的部分
        checkpoint = torch.load(
            f"{checkpoint_path}/rank_{rank}.pt", 
            map_location="cpu"
        )
    else:
        # 单进程加载完整检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 加载权重
    model.load_state_dict(checkpoint)
    print(f"Model loaded from {checkpoint_path}")

def main():
    """主函数，解析命令行参数并执行模型加载和推理"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run Transformer model inference")
    parser.add_argument("--model-size", type=str, default="7b", choices=["7b", "13b", "70b"],
                        help="Model size configuration")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp8"],
                        help="Computation precision")
    parser.add_argument("--use-moe", action="store_true", default=True,
                        help="Use Mixture of Experts architecture")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--prompt", type=str, default="Hello, I am a language model",
                        help="Text prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling threshold")
                        
    args = parser.parse_args()
    
    # 初始化分布式环境
    initialize_distributed()
    world_size, rank = get_distributed_info()
    
    # 设置模型参数
    model_args = setup_model_args(
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        dtype=args.precision,
        use_moe=args.use_moe
    )
    
    # 创建模型
    print(f"Initializing {args.model_size} model with {'MoE' if args.use_moe else 'dense'} architecture")
    model = Transformer(model_args).to(args.device)
    
    # 加载检查点
    load_model(model, args.checkpoint)
    
    # 使用模型生成文本示例
    if rank == 0:  # 只在主进程生成文本
        # 这里应该有一个tokenizer将文本转换为tokens，简化起见，我们使用随机token
        # 真实代码应使用适当的tokenizer
        print(f"Generating text from prompt: {args.prompt}")
        
        # 模拟tokenization
        prompt_tokens = torch.randint(
            0, model_args.vocab_size, (1, len(args.prompt.split())), 
            device=args.device
        )
        
        # 计时生成过程
        start_time = time.time()
        
        # 生成文本
        output_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        end_time = time.time()
        
        # 在真实代码中，应该将tokens解码为文本
        print(f"Generated {args.max_new_tokens} tokens in {end_time - start_time:.2f} seconds")
        print(f"Generation speed: {args.max_new_tokens / (end_time - start_time):.2f} tokens/sec")
        
        # 模拟输出文本
        print(f"Generated text would appear here (requires actual tokenizer for decoding)")
    
    print("Done!")

if __name__ == "__main__":
    main()