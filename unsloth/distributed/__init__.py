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
Unsloth Distributed Training Module

This module extends Unsloth's single-GPU optimizations to multi-GPU setups,
providing support for:
- DDP, FSDP, and DeepSpeed ZeRO training strategies
- Distributed LoRA parameter and optimizer state sharding  
- Multi-GPU FlashAttention and fused kernel support
- CPU/NVMe parameter and gradient offloading
- Distributed checkpointing and model loading
"""

import os
import warnings
from typing import Optional, Dict, Any, List, Union
import torch
import torch.distributed as dist
from packaging.version import Version

# Import distributed components
from .device_manager import DistributedDeviceManager
from .distributed_setup import DistributedEnvironmentSetup
from .model_parallel import UnslothDistributedModel
from .utils import (
    is_distributed_available,
    get_distributed_backend,
    setup_distributed_training,
    cleanup_distributed,
    DistributedConfig,
)

__all__ = [
    "DistributedDeviceManager", 
    "DistributedEnvironmentSetup",
    "UnslothDistributedModel",
    "DistributedConfig",
    "is_distributed_available",
    "get_distributed_backend", 
    "setup_distributed_training",
    "cleanup_distributed",
    "enable_distributed_training",
    "UnslothDistributedTrainer",
]

def enable_distributed_training() -> bool:
    """
    Enable distributed training capabilities for Unsloth.
    
    Returns:
        bool: True if distributed training was successfully enabled
    """
    if not is_distributed_available():
        warnings.warn(
            "Distributed training packages not available. "
            "Install accelerate, deepspeed, or fairscale for multi-GPU support."
        )
        return False
    
    # Set environment variable to indicate distributed mode
    os.environ["UNSLOTH_DISTRIBUTED_ENABLED"] = "1"
    
    # Initialize distributed backend if not already done
    if not dist.is_initialized():
        try:
            setup_distributed_training()
            return True
        except Exception as e:
            warnings.warn(f"Failed to initialize distributed training: {e}")
            return False
    
    return True

# Lazy import of trainer to avoid circular dependencies
def _get_distributed_trainer():
    from .trainer import UnslothDistributedTrainer
    return UnslothDistributedTrainer

# Make trainer available through lazy loading
import sys
def __getattr__(name):
    if name == "UnslothDistributedTrainer":
        return _get_distributed_trainer()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
