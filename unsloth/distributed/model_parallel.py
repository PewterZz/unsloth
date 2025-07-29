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
Distributed model parallel implementation for Unsloth
Handles LoRA parameter sharding and distributed forward/backward passes
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist

# LoRA and PEFT imports
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft.tuners.lora import LoraLayer
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# FlashAttn imports
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from .utils import (
    DistributedConfig,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    reduce_dict,
    barrier,
)

@dataclass
class DistributedLoRAConfig:
    """Configuration for distributed LoRA training"""
    
    r: int = 16
    lora_alpha: int = 32
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.1
    shard_strategy: str = "hybrid"  # "parameter", "rank", "hybrid"
    enable_gradient_checkpointing: bool = True
    use_rslora: bool = False
    use_dora: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        if self.shard_strategy not in ["parameter", "rank", "hybrid"]:
            raise ValueError(f"Invalid shard_strategy: {self.shard_strategy}")

class DistributedLoRALayer(nn.Module):
    """
    Distributed LoRA layer that shards parameters across GPUs
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.1,
        shard_strategy: str = "hybrid",
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.shard_strategy = shard_strategy
        self.world_size = world_size
        self.rank = rank
        
        # Get base layer dimensions
        if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        elif hasattr(base_layer, "weight"):
            self.out_features, self.in_features = base_layer.weight.shape
        else:
            raise ValueError("Cannot determine layer dimensions")
        
        # Initialize LoRA parameters based on sharding strategy
        self._init_distributed_lora_params()
        
        # Dropout layer
        self.lora_dropout_layer = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
    def _init_distributed_lora_params(self):
        """Initialize LoRA parameters with appropriate sharding"""
        
        if self.shard_strategy == "parameter":
            # Shard A and B matrices across different GPUs
            self._init_parameter_sharding()
        elif self.shard_strategy == "rank":
            # Shard rank dimension across GPUs
            self._init_rank_sharding()
        elif self.shard_strategy == "hybrid":
            # Use hybrid approach: rank sharding for large layers, parameter for small
            if self.r >= 32 and self.world_size >= 4:
                self._init_rank_sharding()
            else:
                self._init_parameter_sharding()
        
    def _init_parameter_sharding(self):
        """Initialize with parameter sharding (different GPUs handle A vs B)"""
        device = next(self.base_layer.parameters()).device
        dtype = next(self.base_layer.parameters()).dtype
        
        # Decide which parameters this GPU should hold
        if self.rank % 2 == 0:
            # Even ranks hold lora_A
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False, device=device, dtype=dtype)
            self.lora_B = None
            # Initialize A with kaiming uniform
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        else:
            # Odd ranks hold lora_B  
            self.lora_A = None
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False, device=device, dtype=dtype)
            # Initialize B to zero
            nn.init.zeros_(self.lora_B.weight)
            
    def _init_rank_sharding(self):
        """Initialize with rank dimension sharding"""
        device = next(self.base_layer.parameters()).device
        dtype = next(self.base_layer.parameters()).dtype
        
        # Calculate sharded rank dimension
        sharded_r = self.r // self.world_size
        if self.rank < self.r % self.world_size:
            sharded_r += 1
            
        # Each GPU holds a slice of both A and B
        self.lora_A = nn.Linear(self.in_features, sharded_r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(sharded_r, self.out_features, bias=False, device=device, dtype=dtype)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        self.sharded_r = sharded_r
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with distributed LoRA computation"""
        # Base layer computation
        base_output = self.base_layer(x)
        
        # LoRA computation based on sharding strategy
        if self.shard_strategy == "parameter":
            lora_output = self._forward_parameter_sharded(x)
        elif self.shard_strategy in ["rank", "hybrid"]:
            lora_output = self._forward_rank_sharded(x)
        else:
            raise ValueError(f"Unknown shard strategy: {self.shard_strategy}")
            
        return base_output + lora_output * self.scaling
    
    def _forward_parameter_sharded(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with parameter sharding"""
        # Apply dropout
        x_dropout = self.lora_dropout_layer(x)
        
        if self.rank % 2 == 0 and self.lora_A is not None:
            # Compute A @ x and send to next rank
            a_output = self.lora_A(x_dropout)  # Shape: [batch, seq, r]
            
            # Send to next rank (which should have lora_B)
            if self.rank + 1 < self.world_size:
                dist.send(a_output, dst=self.rank + 1)
                
            # Receive B @ (A @ x) from next rank
            if self.rank + 1 < self.world_size:
                final_output = torch.zeros(x_dropout.shape[0], x_dropout.shape[1], self.out_features,
                                         device=x_dropout.device, dtype=x_dropout.dtype)
                dist.recv(final_output, src=self.rank + 1)
                return final_output
            else:
                return torch.zeros(x_dropout.shape[0], x_dropout.shape[1], self.out_features,
                                 device=x_dropout.device, dtype=x_dropout.dtype)
                
        elif self.rank % 2 == 1 and self.lora_B is not None:
            # Receive A @ x from previous rank
            a_output = torch.zeros(x_dropout.shape[0], x_dropout.shape[1], self.r, 
                                 device=x_dropout.device, dtype=x_dropout.dtype)
            if self.rank - 1 >= 0:
                dist.recv(a_output, src=self.rank - 1)
            
            # Compute B @ (A @ x) 
            final_output = self.lora_B(a_output)
            
            # Send result back to previous rank
            if self.rank - 1 >= 0:
                dist.send(final_output, dst=self.rank - 1)
            
            return final_output
        
        return torch.zeros(x_dropout.shape[0], x_dropout.shape[1], self.out_features,
                         device=x_dropout.device, dtype=x_dropout.dtype)
    
    def _forward_rank_sharded(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with rank sharding"""
        # Apply dropout
        x_dropout = self.lora_dropout_layer(x)
        
        # Each GPU computes its portion: B_i @ A_i @ x
        if self.lora_A is not None and self.lora_B is not None:
            a_output = self.lora_A(x_dropout)  # Shape: [batch, seq, sharded_r]
            local_output = self.lora_B(a_output)  # Shape: [batch, seq, out_features]
        else:
            local_output = torch.zeros(x_dropout.shape[0], x_dropout.shape[1], self.out_features,
                                     device=x_dropout.device, dtype=x_dropout.dtype)
        
        # All-reduce to sum contributions from all GPUs
        if self.world_size > 1:
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
            
        return local_output

class UnslothDistributedModel(nn.Module):
    """
    Main distributed model wrapper that integrates Unsloth optimizations 
    with multi-GPU training capabilities
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        distributed_config: DistributedConfig,
        lora_config: Optional[DistributedLoRAConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.distributed_config = distributed_config
        self.lora_config = lora_config
        
        self.world_size = get_world_size()
        self.rank = get_rank()
        
        # Apply distributed LoRA if configured
        if lora_config is not None and distributed_config.distribute_lora:
            self._apply_distributed_lora()
        
        # Setup gradient checkpointing
        if distributed_config.gradient_checkpointing:
            self._setup_gradient_checkpointing()
    
    def _apply_distributed_lora(self):
        """Apply distributed LoRA to target modules"""
        if not HAS_PEFT:
            warnings.warn("PEFT not available. Skipping LoRA application.")
            return
            
        target_modules = self.lora_config.target_modules
        
        # Find and replace target modules
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with distributed LoRA layer
                    distributed_lora = DistributedLoRALayer(
                        base_layer=module,
                        r=self.lora_config.r,
                        lora_alpha=self.lora_config.lora_alpha,
                        lora_dropout=self.lora_config.lora_dropout,
                        shard_strategy=self.lora_config.shard_strategy,
                        world_size=self.world_size,
                        rank=self.rank,
                    )
                    
                    # Replace module in parent
                    parent_name, child_name = name.rsplit('.', 1)
                    parent_module = self.base_model.get_submodule(parent_name)
                    setattr(parent_module, child_name, distributed_lora)
    
    def _setup_gradient_checkpointing(self):
        """Setup distributed gradient checkpointing"""
        def checkpoint_wrapper(module):
            def checkpoint_forward(*args, **kwargs):
                def run_function(*inputs):
                    return module(*inputs, **kwargs)
                return torch.utils.checkpoint.checkpoint(run_function, *args)
            return checkpoint_forward
        
        # Apply checkpointing to transformer layers
        for name, module in self.base_model.named_modules():
            if "layer" in name.lower() or "block" in name.lower():
                if hasattr(module, "forward"):
                    module.forward = checkpoint_wrapper(module)
    
    def forward(self, *args, **kwargs):
        """Forward pass with distributed optimizations"""
        return self.base_model(*args, **kwargs)
    
    def get_distributed_parameters(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Get parameters grouped by distribution strategy"""
        params = {
            "base_model": [],
            "lora_A": [],
            "lora_B": [],
            "shared": [],
        }
        
        for name, param in self.named_parameters():
            if "lora_A" in name:
                params["lora_A"].append(param)
            elif "lora_B" in name:
                params["lora_B"].append(param)
            elif any(x in name for x in ["embed", "norm", "head"]):
                params["shared"].append(param)
            else:
                params["base_model"].append(param)
                
        return params
    
    def sync_distributed_parameters(self):
        """Synchronize distributed parameters across ranks"""
        if self.world_size == 1:
            return
            
        # Synchronize shared parameters (embeddings, norms, etc.)
        param_groups = self.get_distributed_parameters()
        
        for param in param_groups["shared"]:
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size
    
    @contextmanager
    def distributed_inference_mode(self):
        """Context manager for efficient distributed inference"""
        was_training = self.training
        
        try:
            self.eval()
            with torch.no_grad():
                yield
        finally:
            if was_training:
                self.train()
    
    def save_distributed_checkpoint(self, checkpoint_path: str, include_optimizer: bool = True):
        """Save distributed model checkpoint"""
        if not is_main_process():
            return
            
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "distributed_config": self.distributed_config,
            "lora_config": self.lora_config,
            "world_size": self.world_size,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Distributed checkpoint saved: {checkpoint_path}")
    
    @classmethod
    def load_distributed_checkpoint(
        cls, 
        checkpoint_path: str,
        base_model: nn.Module,
    ) -> 'UnslothDistributedModel':
        """Load distributed model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Create distributed model
        distributed_model = cls(
            base_model=base_model,
            distributed_config=checkpoint["distributed_config"],
            lora_config=checkpoint.get("lora_config"),
        )
        
        # Load state dict
        distributed_model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"Distributed checkpoint loaded: {checkpoint_path}")
        return distributed_model
