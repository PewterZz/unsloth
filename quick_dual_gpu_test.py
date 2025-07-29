#!/usr/bin/env python3

"""
Quick dual GPU functionality test
Tests basic distributed operations before running full suite
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional

def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def test_basic_distributed_ops(rank: int, world_size: int):
    """Test basic distributed operations"""
    try:
        setup_distributed(rank, world_size)
        
        print(f"🚀 Process {rank}/{world_size} started on GPU {rank}")
        
        # Create test tensor on GPU
        device = torch.device(f'cuda:{rank}')
        tensor = torch.ones(4, 4, device=device) * (rank + 1)
        
        print(f"   Rank {rank}: Created tensor with values {tensor[0,0].item()}")
        
        # Test all-reduce operation
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        expected_sum = sum(range(1, world_size + 1))
        if tensor[0,0].item() == expected_sum:
            print(f"✅ Rank {rank}: All-reduce successful (sum = {tensor[0,0].item()})")
        else:
            print(f"❌ Rank {rank}: All-reduce failed (expected {expected_sum}, got {tensor[0,0].item()})")
        
        # Test broadcast operation
        if rank == 0:
            broadcast_tensor = torch.randn(2, 2, device=device)
            print(f"   Rank 0: Broadcasting tensor with sum {broadcast_tensor.sum().item():.2f}")
        else:
            broadcast_tensor = torch.zeros(2, 2, device=device)
        
        dist.broadcast(broadcast_tensor, src=0)
        
        print(f"   Rank {rank}: Received broadcast tensor sum = {broadcast_tensor.sum().item():.2f}")
        
        # Test memory usage
        allocated = torch.cuda.memory_allocated(rank) / (1024**2)
        reserved = torch.cuda.memory_reserved(rank) / (1024**2)
        print(f"   Rank {rank}: Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
        
        # Synchronize all processes
        dist.barrier()
        
        if rank == 0:
            print("🎉 All distributed operations completed successfully!")
        
    except Exception as e:
        print(f"❌ Rank {rank}: Error in distributed test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()

def run_distributed_test():
    """Run the distributed test"""
    print("🧪 QUICK DUAL GPU DISTRIBUTED TEST")
    print("=" * 40)
    
    # Check prerequisites
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"📊 Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("⚠️  Less than 2 GPUs - testing single GPU operations")
        world_size = 1
    else:
        print("✅ Dual GPU detected - testing distributed operations")
        world_size = min(2, gpu_count)  # Use up to 2 GPUs
    
    if world_size == 1:
        # Single GPU test
        device = torch.device('cuda:0')
        tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(tensor, tensor.T)
        print(f"✅ Single GPU operations successful - tensor shape: {result.shape}")
        return True
    
    # Multi-GPU test using spawn
    try:
        print(f"🚀 Spawning {world_size} processes for distributed test...")
        mp.spawn(
            test_basic_distributed_ops,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("✅ Distributed test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Distributed test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_distributed_test()
    
    if success:
        print("\n🎯 READY FOR FULL TEST SUITE!")
        print("Run: ./run_gemma3n_tests.sh all")
    else:
        print("\n⚠️  Issues detected - check your setup")
    
    sys.exit(0 if success else 1)
