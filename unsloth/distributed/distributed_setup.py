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
Distributed environment setup and configuration for Unsloth
"""

import os
import json
import warnings
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import distributed backends
try:
    import accelerate
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        BackwardPrefetch,
        MixedPrecision,
        ShardingStrategy,
    )
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False

from .utils import (
    DistributedConfig,
    DistributedBackend,
    is_main_process,
    get_world_size,
    get_rank,
    get_local_rank,
)

class DistributedEnvironmentSetup:
    """
    Setup and manage distributed training environment for Unsloth
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.accelerator = None
        self.wrapped_model = None
        self.backend_config = {}
        
    def setup_accelerator(self) -> Optional[Accelerator]:
        """Setup Accelerate for distributed training"""
        if not HAS_ACCELERATE:
            warnings.warn("Accelerate not available. Install with: pip install accelerate")
            return None
            
        # Create accelerator config
        accelerator_config = {
            "mixed_precision": "bf16" if self.config.mixed_precision else "no",
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "dispatch_batches": False,
            "split_batches": True,
        }
        
        # Add DeepSpeed config if using DeepSpeed
        if self.config.backend.value.startswith("deepspeed"):
            accelerator_config["deepspeed_plugin"] = self._create_deepspeed_config()
        
        self.accelerator = Accelerator(**accelerator_config)
        
        if is_main_process():
            print(f"Accelerator initialized with {self.accelerator.num_processes} processes")
            
        return self.accelerator
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration"""
        zero_stage = self.config.zero_stage
        if self.config.backend == DistributedBackend.DEEPSPEED_ZERO1:
            zero_stage = 1
        elif self.config.backend == DistributedBackend.DEEPSPEED_ZERO2:
            zero_stage = 2
        elif self.config.backend == DistributedBackend.DEEPSPEED_ZERO3:
            zero_stage = 3
            
        deepspeed_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "zero_optimization": {
                "stage": zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": self.config.bucket_cap_mb * 1024 * 1024,
                "overlap_comm": self.config.overlap_communication,
                "reduce_scatter": True,
                "reduce_bucket_size": self.config.bucket_cap_mb * 1024 * 1024,
                "contiguous_gradients": True,
            },
            "fp16": {
                "enabled": self.config.mixed_precision and not torch.cuda.is_bf16_supported(),
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.config.mixed_precision and torch.cuda.is_bf16_supported(),
            },
            "activation_checkpointing": {
                "partition_activations": self.config.activation_checkpointing,
                "cpu_checkpointing": self.config.cpu_offload,
                "contiguous_memory_optimization": True,
                "number_checkpoints": 4,
            },
            "wall_clock_breakdown": False,
        }
        
        # Add CPU/NVMe offloading
        if self.config.cpu_offload:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {
                "device": self.config.offload_optimizer_device,
                "pin_memory": True,
            }
            
            if zero_stage == 3:
                deepspeed_config["zero_optimization"]["offload_param"] = {
                    "device": self.config.offload_param_device,
                    "pin_memory": True,
                }
        
        return deepspeed_config
    
    def wrap_model_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel"""
        if get_world_size() == 1:
            return model
            
        # Move model to current device
        device = torch.cuda.current_device()
        model = model.to(device)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            bucket_cap_mb=self.config.bucket_cap_mb,
        )
        
        if is_main_process():
            print("Model wrapped with DistributedDataParallel")
            
        return ddp_model
    
    def wrap_model_fsdp(self, model: nn.Module) -> nn.Module:
        """Wrap model with FullyShardedDataParallel"""
        if not HAS_FSDP:
            warnings.warn("FSDP not available. Falling back to DDP")
            return self.wrap_model_ddp(model)
            
        if get_world_size() == 1:
            return model
        
        # Configure mixed precision
        mixed_precision_policy = None
        if self.config.mixed_precision:
            if torch.cuda.is_bf16_supported():
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            else:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
        
        # Configure auto-wrap policy
        def lambda_auto_wrap_policy(module, recurse, nonwrapped_numel):
            if recurse:
                return True
            return nonwrapped_numel >= 1e8  # 100M parameters
        
        # Configure sharding strategy
        sharding_strategy = ShardingStrategy.FULL_SHARD
        if not self.config.shard_parameters:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            
        # Wrap model with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=lambda_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE if self.config.overlap_communication else None,
            device_id=torch.cuda.current_device(),
            sync_module_states=self.config.sync_module_states,
            use_orig_params=self.config.use_orig_params,
        )
        
        if is_main_process():
            print("Model wrapped with FullyShardedDataParallel")
            
        return fsdp_model
    
    def wrap_model_deepspeed(
        self, 
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> nn.Module:
        """Wrap model with DeepSpeed"""
        if not HAS_DEEPSPEED:
            warnings.warn("DeepSpeed not available. Falling back to FSDP")
            return self.wrap_model_fsdp(model)
            
        # Create DeepSpeed config
        ds_config = self._create_deepspeed_config()
        
        # Initialize DeepSpeed engine
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
        )
        
        if is_main_process():
            print(f"Model wrapped with DeepSpeed ZeRO Stage {ds_config['zero_optimization']['stage']}")
            
        return model_engine
    
    def wrap_model(
        self, 
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> nn.Module:
        """
        Wrap model with appropriate distributed training backend
        """
        if self.config.backend == DistributedBackend.DDP:
            wrapped_model = self.wrap_model_ddp(model)
        elif self.config.backend == DistributedBackend.FSDP:
            wrapped_model = self.wrap_model_fsdp(model)
        elif self.config.backend.value.startswith("deepspeed"):
            wrapped_model = self.wrap_model_deepspeed(model, optimizer, lr_scheduler)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        
        self.wrapped_model = wrapped_model
        return wrapped_model
    
    def setup_distributed_sampler(self, dataset, shuffle: bool = True):
        """Setup distributed data sampler"""
        if get_world_size() == 1:
            return None
            
        from torch.utils.data.distributed import DistributedSampler
        
        return DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
    
    def get_model_size_gb(self, model: nn.Module) -> float:
        """Estimate model size in GB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
            
        total_size = param_size + buffer_size
        return total_size / (1024**3)  # Convert to GB
    
    def print_model_info(self, model: nn.Module):
        """Print model information for distributed training"""
        if not is_main_process():
            return
            
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size
        model_size_gb = self.get_model_size_gb(model)
        
        print("\n" + "="*60)
        print("Distributed Model Information")
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_gb:.2f} GB")
        print(f"Backend: {self.config.backend.value}")
        print(f"World size: {get_world_size()}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"Gradient checkpointing: {self.config.gradient_checkpointing}")
        print(f"CPU offload: {self.config.cpu_offload}")
        print("="*60 + "\n")
