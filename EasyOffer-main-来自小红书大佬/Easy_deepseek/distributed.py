"""
分布式训练支持模块 - 提供分布式环境的初始化与管理功能。
该模块处理多GPU/多节点环境中的通信和同步逻辑。
"""

import torch
import torch.distributed as dist

# 全局分布式环境变量
world_size = 1  # 分布式训练中的总进程数
rank = 0        # 当前进程在分布式环境中的排名(ID)

def initialize_distributed(backend: str = "nccl") -> tuple[int, int]:
    """
    初始化分布式训练环境并设置全局变量。
    
    参数:
        backend (str): 分布式后端类型，对GPU使用'nccl'，对CPU使用'gloo'
        
    返回:
        tuple[int, int]: (world_size, rank) 总进程数和当前进程ID
    """
    global world_size, rank
    
    if dist.is_available() and dist.is_initialized():
        # 分布式环境已初始化，直接获取参数
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        # 尝试初始化分布式环境
        if torch.cuda.is_available():
            try:
                # 假设环境变量(如MASTER_ADDR, MASTER_PORT等)已正确设置
                dist.init_process_group(backend=backend)
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            except Exception as e:
                print(f"无法初始化分布式环境: {e}")
                print("使用单进程模式运行")
                world_size = 1
                rank = 0
        else:
            # 无GPU或分布式设置，使用单进程
            world_size = 1
            rank = 0
            
    return world_size, rank

def get_distributed_info() -> tuple[int, int]:
    """
    获取当前分布式环境信息。
    
    返回:
        tuple[int, int]: (world_size, rank) 总进程数和当前进程ID
    """
    return world_size, rank

def update_distributed_info(new_world_size: int, new_rank: int):
    """
    手动更新分布式环境信息(通常用于测试或特殊配置)。
    
    参数:
        new_world_size (int): 新的进程总数
        new_rank (int): 新的当前进程ID
    """
    global world_size, rank
    world_size = new_world_size
    rank = new_rank