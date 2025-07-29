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
Device management for distributed Unsloth training
"""

import os
import warnings
from typing import Dict, List, Optional, Union, Any
import torch
import torch.distributed as dist
from contextlib import contextmanager

# Try to import psutil and GPUtil for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

from .utils import (
    get_world_size, 
    get_rank, 
    get_local_rank,
    is_main_process,
    DistributedConfig
)

class DistributedDeviceManager:
    """
    Manages device allocation and memory optimization for distributed training
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        
        # Device information
        self.device_count = torch.cuda.device_count()
        self.current_device = torch.cuda.current_device()
        
        # Memory tracking
        self.memory_stats = {}
        self.max_memory_allocated = 0
        self.peak_memory_reserved = 0
        
        # Initialize device
        self._setup_device()
        self._setup_memory_management()
    
    def _setup_device(self):
        """Setup CUDA device for current process"""
        if torch.cuda.is_available():
            # Set device based on local rank
            device_id = self.local_rank % self.device_count
            torch.cuda.set_device(device_id)
            self.current_device = device_id
            
            if is_main_process():
                print(f"Distributed training on {self.world_size} GPUs")
                print(f"Process {self.rank} using GPU {device_id}/{self.device_count}")
        else:
            warnings.warn("CUDA not available. Distributed training may be slow.")
    
    def _setup_memory_management(self):
        """Configure memory management for distributed training"""
        if not torch.cuda.is_available():
            return
            
        # Configure memory allocator for distributed training
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,"
            "roundup_power2_divisions:[32:256,64:128,256:64,>:1],"
            "garbage_collection_threshold:0.6"
        )
        
        # Set memory fraction if specified
        if self.config.max_memory_per_gpu:
            try:
                memory_fraction = self._parse_memory_string(self.config.max_memory_per_gpu)
                torch.cuda.set_per_process_memory_fraction(memory_fraction, self.current_device)
            except Exception as e:
                warnings.warn(f"Failed to set memory fraction: {e}")
    
    def _parse_memory_string(self, memory_str: str) -> float:
        """Parse memory string like '8GB' or '0.8' to fraction"""
        if memory_str.endswith("GB"):
            # Convert GB to fraction of total GPU memory
            memory_gb = float(memory_str[:-2])
            total_memory = torch.cuda.get_device_properties(self.current_device).total_memory
            total_gb = total_memory / (1024**3)
            return min(memory_gb / total_gb, 1.0)
        elif memory_str.endswith("MB"):
            # Convert MB to fraction
            memory_mb = float(memory_str[:-2])
            total_memory = torch.cuda.get_device_properties(self.current_device).total_memory
            total_mb = total_memory / (1024**2)
            return min(memory_mb / total_mb, 1.0)
        else:
            # Assume it's already a fraction
            return min(float(memory_str), 1.0)
    
    def get_device_map(self, model_size_gb: Optional[float] = None) -> Dict[str, Union[int, str]]:
        """
        Generate optimal device map for model sharding
        
        Args:
            model_size_gb: Estimated model size in GB
            
        Returns:
            Device map for model placement
        """
        if self.world_size == 1:
            return {"": self.current_device}
        
        # For multi-GPU, use automatic sharding
        if model_size_gb is None:
            # Use sequential placement
            return {"": "sequential"}
        
        # Estimate memory requirements
        available_memory = self._get_available_memory_per_gpu()
        
        # If model fits on single GPU, place everything there
        if model_size_gb * 1.5 < available_memory:  # 1.5x for overhead
            return {"": self.current_device}
        
        # Otherwise, use balanced sharding
        return {"": "balanced"}
    
    def _get_available_memory_per_gpu(self) -> float:
        """Get available memory per GPU in GB"""
        if not torch.cuda.is_available():
            return 0.0
            
        try:
            device_props = torch.cuda.get_device_properties(self.current_device)
            total_memory = device_props.total_memory / (1024**3)  # Convert to GB
            
            # Reserve some memory for PyTorch overhead
            reserved_fraction = 0.1  # 10% reserved
            available_memory = total_memory * (1 - reserved_fraction)
            
            return available_memory
        except Exception:
            return 4.0  # Default conservative estimate
    
    def get_cpu_memory_gb(self) -> float:
        """Get available CPU memory in GB"""
        if HAS_PSUTIL:
            try:
                return psutil.virtual_memory().available / (1024**3)
            except Exception:
                pass
        return 8.0  # Default estimate
    
    def optimize_for_distributed_training(self):
        """Apply optimizations for distributed training"""
        if not torch.cuda.is_available():
            return
            
        # Enable memory pool for faster allocation
        torch.cuda.empty_cache()
        
        # Set optimal CUDA settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Enable compilation optimizations
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.cache_size_limit = 1000
            torch._dynamo.config.optimize_ddp = True
    
    @contextmanager
    def memory_profiler(self, tag: str = ""):
        """Context manager for memory profiling"""
        if not torch.cuda.is_available():
            yield
            return
            
        # Record starting memory
        torch.cuda.synchronize()
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
        
        try:
            yield
        finally:
            # Record ending memory
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            
            # Calculate differences
            allocated_diff = end_allocated - start_allocated
            reserved_diff = end_reserved - start_reserved
            
            # Store stats
            self.memory_stats[tag] = {
                "allocated_mb": allocated_diff / (1024**2),
                "reserved_mb": reserved_diff / (1024**2),
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
            }
            
            if is_main_process() and tag:
                print(f"Memory usage for {tag}:")
                print(f"  Allocated: {allocated_diff/(1024**2):.1f} MB")
                print(f"  Reserved: {reserved_diff/(1024**2):.1f} MB")
    
    def print_memory_summary(self):
        """Print memory usage summary"""
        if not torch.cuda.is_available() or not is_main_process():
            return
            
        current_allocated = torch.cuda.memory_allocated() / (1024**2)
        current_reserved = torch.cuda.memory_reserved() / (1024**2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        
        print("\n" + "="*50)
        print("GPU Memory Summary")
        print("="*50)
        print(f"Current Allocated: {current_allocated:.1f} MB")
        print(f"Current Reserved:  {current_reserved:.1f} MB")
        print(f"Peak Allocated:    {max_allocated:.1f} MB")
        print(f"Peak Reserved:     {max_reserved:.1f} MB")
        
        # Show per-operation stats
        if self.memory_stats:
            print("\nPer-operation memory usage:")
            for tag, stats in self.memory_stats.items():
                print(f"  {tag}: {stats['allocated_mb']:.1f} MB allocated")
        
        print("="*50 + "\n")
    
    def clear_memory(self):
        """Clear GPU memory caches"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
    
    def get_optimal_batch_size(
        self, 
        base_batch_size: int,
        model_size_gb: float,
        sequence_length: int = 2048,
    ) -> int:
        """
        Calculate optimal batch size for distributed training
        
        Args:
            base_batch_size: Base batch size for single GPU
            model_size_gb: Model size in GB
            sequence_length: Sequence length
            
        Returns:
            Optimal batch size per GPU
        """
        available_memory = self._get_available_memory_per_gpu()
        
        # Estimate memory usage per sample (rough approximation)
        # Formula: model_size + (batch_size * seq_len * hidden_size * 4 bytes * 2) / 1GB
        hidden_size = 4096  # Rough estimate
        memory_per_sample = (sequence_length * hidden_size * 4 * 2) / (1024**3)  # GB
        
        # Available memory for batch processing
        available_for_batch = available_memory - model_size_gb - 1.0  # 1GB buffer
        
        if available_for_batch <= 0:
            return 1
        
        # Calculate max batch size that fits in memory
        max_batch_size = int(available_for_batch / memory_per_sample)
        
        # Use smaller of base batch size and max batch size
        optimal_batch_size = min(base_batch_size, max_batch_size, 32)  # Cap at 32
        
        return max(1, optimal_batch_size)
    
    def setup_cpu_offload(self):
        """Setup CPU offloading for parameters and optimizer states"""
        if not self.config.cpu_offload:
            return None
            
        # Check available CPU memory
        cpu_memory_gb = self.get_cpu_memory_gb()
        
        if cpu_memory_gb < 8.0:  # Minimum 8GB RAM required
            warnings.warn("Insufficient CPU memory for offloading. Disabling CPU offload.")
            return None
        
        offload_config = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4,
            "buffer_size": int(cpu_memory_gb * 0.2 * 1024**3),  # 20% of available RAM
        }
        
        if is_main_process():
            print(f"CPU offload enabled with {cpu_memory_gb:.1f}GB available RAM")
        
        return offload_config
    
    def setup_nvme_offload(self, offload_dir: str = "/tmp/unsloth_offload"):
        """Setup NVMe offloading for optimizer states"""
        if not self.config.nvme_offload:
            return None
            
        # Create offload directory
        os.makedirs(offload_dir, exist_ok=True)
        
        # Check available disk space
        if HAS_PSUTIL:
            try:
                disk_usage = psutil.disk_usage(offload_dir)
                available_gb = disk_usage.free / (1024**3)
                
                if available_gb < 10.0:  # Minimum 10GB required
                    warnings.warn("Insufficient disk space for NVMe offload. Disabling.")
                    return None
                    
                offload_config = {
                    "device": "nvme",
                    "nvme_path": offload_dir,
                    "buffer_count": 2,
                    "buffer_size": min(int(available_gb * 0.1 * 1024**3), 2**31-1),  # 10% of space, max 2GB
                }
                
                if is_main_process():
                    print(f"NVMe offload enabled with {available_gb:.1f}GB available space")
                
                return offload_config
                
            except Exception as e:
                warnings.warn(f"Failed to setup NVMe offload: {e}")
                return None
        else:
            warnings.warn("psutil not available. Cannot setup NVMe offload.")
            return None

# Alias for backward compatibility
DeviceManager = DistributedDeviceManager
