# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Distributed training utilities for Unsloth
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Literal
from enum import Enum

import torch
import torch.distributed as dist

# Check for distributed packages
try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    import fairscale
    HAS_FAIRSCALE = True
except ImportError:
    HAS_FAIRSCALE = False

class DistributedBackend(Enum):
    """Supported distributed training backends"""
    DDP = "ddp"
    FSDP = "fsdp" 
    DEEPSPEED = "deepspeed"
    DEEPSPEED_ZERO1 = "deepspeed_zero1"
    DEEPSPEED_ZERO2 = "deepspeed_zero2"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    
    # Backend configuration
    backend: DistributedBackend = DistributedBackend.FSDP
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Model sharding
    shard_parameters: bool = True
    shard_gradients: bool = True
    shard_optimizer_states: bool = True
    
    # Memory optimization
    cpu_offload: bool = False
    nvme_offload: bool = False
    offload_params: bool = True
    offload_optimizer: bool = True
    
    # LoRA specific settings
    distribute_lora: bool = True
    lora_shard_strategy: Literal["parameter", "rank", "hybrid"] = "hybrid"
    
    # Communication settings
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    
    # Performance tuning
    overlap_communication: bool = True
    use_orig_params: bool = True
    sync_module_states: bool = True
    
    # DeepSpeed specific
    zero_stage: int = 2
    offload_param_device: str = "cpu"
    offload_optimizer_device: str = "cpu"
    zero_force_ds_cpu_optimizer: bool = True
    
    # Advanced settings
    activation_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    max_memory_per_gpu: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration"""
        if self.backend == DistributedBackend.DEEPSPEED and not HAS_DEEPSPEED:
            raise ImportError("DeepSpeed not available. Install with: pip install deepspeed")
            
        if self.backend == DistributedBackend.FSDP and not HAS_FAIRSCALE:
            warnings.warn("FairScale not available. Falling back to PyTorch native FSDP")
            
        if self.nvme_offload and not self.cpu_offload:
            warnings.warn("NVMe offload requires CPU offload. Enabling CPU offload.")
            self.cpu_offload = True

def is_distributed_available() -> bool:
    """Check if distributed training packages are available"""
    return HAS_ACCELERATE or HAS_DEEPSPEED or HAS_FAIRSCALE

def get_distributed_backend() -> Optional[str]:
    """Get the best available distributed backend"""
    if HAS_DEEPSPEED:
        return "deepspeed"
    elif HAS_FAIRSCALE:
        return "fsdp"
    elif HAS_ACCELERATE:
        return "ddp"
    else:
        return None

def setup_distributed_training(
    backend: Optional[str] = None,
    init_method: str = "env://",
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    timeout_minutes: int = 30,
) -> bool:
    """
    Initialize distributed training environment
    
    Args:
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method
        world_size: Total number of processes
        rank: Rank of current process
        timeout_minutes: Timeout for initialization
        
    Returns:
        bool: True if successfully initialized
    """
    if dist.is_initialized():
        return True
        
    # Auto-detect environment variables
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
        
    # Select backend
    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
    
    try:
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.default_pg_timeout * timeout_minutes,
        )
        
        # Set CUDA device for current process
        if torch.cuda.is_available() and backend == "nccl":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to initialize distributed training: {e}")
        return False

def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_world_size() -> int:
    """Get total number of processes"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank() -> int:
    """Get rank of current process"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_local_rank() -> int:
    """Get local rank of current process"""
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_main_process() -> bool:
    """Check if current process is main process"""
    return get_rank() == 0

def barrier():
    """Synchronize all processes""" 
    if dist.is_initialized():
        dist.barrier()

def broadcast_object(obj: Any, src: int = 0):
    """Broadcast object from source rank to all ranks"""
    if dist.is_initialized():
        dist.broadcast_object_list([obj], src=src)
        return obj
    return obj

def all_gather_object(obj: Any) -> List[Any]:
    """Gather objects from all ranks"""
    if not dist.is_initialized():
        return [obj]
        
    world_size = get_world_size()
    gathered_objects = [None] * world_size
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects

def reduce_dict(input_dict: Dict[str, torch.Tensor], 
                op: dist.ReduceOp = dist.ReduceOp.SUM) -> Dict[str, torch.Tensor]:
    """Reduce dictionary of tensors across all processes"""
    if not dist.is_initialized():
        return input_dict
        
    world_size = get_world_size()
    if world_size == 1:
        return input_dict
        
    # Convert dict to list of tensors
    keys = sorted(input_dict.keys())
    tensors = [input_dict[k].detach().clone() for k in keys]
    
    # Reduce tensors
    for tensor in tensors:
        dist.all_reduce(tensor, op=op)
        if op == dist.ReduceOp.SUM:
            tensor.div_(world_size)
    
    # Convert back to dict
    return {k: t for k, t in zip(keys, tensors)}
