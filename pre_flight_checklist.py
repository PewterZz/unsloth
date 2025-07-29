#!/usr/bin/env python3

"""
Pre-flight checklist for dual GPU distributed training
Run this before executing the main test suite
"""

import os
import sys
import subprocess
import torch

def check_gpu_environment():
    """Comprehensive GPU environment check"""
    print("🔍 DUAL GPU PRE-FLIGHT CHECKLIST")
    print("=" * 50)
    
    # 1. CUDA availability
    print("\n1. CUDA Environment:")
    if not torch.cuda.is_available():
        print("❌ CUDA not available - check NVIDIA drivers")
        return False
    
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
    
    # 2. GPU count and specs
    print("\n2. GPU Detection:")
    gpu_count = torch.cuda.device_count()
    print(f"📊 GPUs detected: {gpu_count}")
    
    if gpu_count < 2:
        print("⚠️  Warning: Less than 2 GPUs detected")
        print("   Distributed training will fall back to single GPU")
    
    # 3. GPU specifications
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"   GPU {i}: {props.name}")
        print(f"           Memory: {memory_gb:.1f} GB")
        print(f"           SM Count: {props.multi_processor_count}")
        
        # Check if it's RTX 4090 or similar high-end GPU
        if "4090" in props.name or memory_gb >= 20:
            print(f"           ✅ High-end GPU detected - optimal for large models")
        else:
            print(f"           ⚠️  Lower memory GPU - may need CPU offloading")
    
    # 4. Memory test
    print("\n3. GPU Memory Test:")
    for i in range(min(gpu_count, 2)):  # Test first 2 GPUs
        torch.cuda.set_device(i)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free = total_memory - reserved
        
        print(f"   GPU {i} Memory:")
        print(f"     Total: {total_memory / (1024**3):.1f} GB")
        print(f"     Free: {free / (1024**3):.1f} GB")
        print(f"     Allocated: {allocated / (1024**2):.1f} MB")
        
        # Test allocation
        try:
            test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
            del test_tensor
            torch.cuda.empty_cache()
            print(f"     ✅ Memory allocation test passed")
        except Exception as e:
            print(f"     ❌ Memory allocation failed: {e}")
    
    # 5. Check distributed capabilities
    print("\n4. Distributed Training Capabilities:")
    
    # Check NCCL
    try:
        import torch.distributed as dist
        print("✅ torch.distributed available")
        
        # Check NCCL backend
        if dist.is_nccl_available():
            print("✅ NCCL backend available (optimal for multi-GPU)")
        else:
            print("⚠️  NCCL not available - will use Gloo backend")
    except ImportError:
        print("❌ torch.distributed not available")
    
    # Check other packages
    packages_to_check = [
        ("accelerate", "Multi-GPU training support"),
        ("peft", "LoRA adapters"),
        ("transformers", "Model loading"),
        ("flash_attn", "Memory-efficient attention (optional)"),
        ("deepspeed", "Advanced distributed training (optional)")
    ]
    
    print("\n5. Package Dependencies:")
    for package, description in packages_to_check:
        try:
            __import__(package)
            version = __import__(package).__version__
            print(f"✅ {package} v{version} - {description}")
        except ImportError:
            if package in ["flash_attn", "deepspeed"]:
                print(f"⚠️  {package} not found - {description}")
            else:
                print(f"❌ {package} REQUIRED - {description}")
    
    # 6. Environment variables
    print("\n6. Environment Variables:")
    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG", 
        "TORCH_DISTRIBUTED_DEBUG",
        "OMP_NUM_THREADS"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"   {var}: {value}")
    
    print("\n" + "=" * 50)
    
    if gpu_count >= 2:
        print("🎉 System ready for dual GPU distributed training!")
        print("\nRecommended next steps:")
        print("1. Run: ./run_gemma3n_tests.sh memory")
        print("2. Run: ./run_gemma3n_tests.sh single")  
        print("3. Run: ./run_gemma3n_tests.sh dual")
        print("4. Run: ./run_gemma3n_tests.sh all")
        return True
    else:
        print("⚠️  Single GPU detected - some tests will be limited")
        return False

if __name__ == "__main__":
    success = check_gpu_environment()
    sys.exit(0 if success else 1)
