#!/usr/bin/env python3

"""
Test suite for Unsloth Distributed Training with Gemma 3N on dual RTX 4090 setup
Optimized for 2x RTX 4090 (24GB VRAM each = 48GB total)
"""

import os
import sys
import torch
import torch.nn as nn
import warnings
from typing import Optional, Dict, Any
from pathlib import Path

# Add unsloth to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our distributed implementation
try:
    from unsloth.distributed import (
        DistributedConfig,
        DistributedBackend,
        DistributedLoRAConfig,
        UnslothDistributedModel,
        DistributedEnvironmentSetup,
        DeviceManager,
        enable_distributed_training,
        get_world_size,
        get_rank,
        is_main_process,
    )
    print("✅ Successfully imported distributed modules")
except ImportError as e:
    print(f"❌ Failed to import distributed modules: {e}")
    sys.exit(1)

# Test configuration for dual RTX 4090
class RTX4090Config:
    """Configuration optimized for dual RTX 4090 setup"""
    
    # RTX 4090 specs
    VRAM_PER_GPU = 24  # GB
    TOTAL_VRAM = 48    # GB
    MEMORY_EFFICIENCY = 0.85  # Reserve 15% for overhead
    
    # Model configurations
    GEMMA_3N_CONFIGS = {
        "gemma-2-9b": {
            "hidden_size": 3584,
            "num_layers": 42,
            "num_attention_heads": 16,
            "intermediate_size": 14336,
            "vocab_size": 256000,
            "estimated_size_gb": 18.0,  # Approximate model size
        },
        "gemma-2-27b": {
            "hidden_size": 4608,
            "num_layers": 46, 
            "num_attention_heads": 32,
            "intermediate_size": 36864,
            "vocab_size": 256000,
            "estimated_size_gb": 54.0,  # Would need CPU offloading
        }
    }
    
    @classmethod
    def get_optimal_batch_size(cls, model_name: str, sequence_length: int = 2048) -> Dict[str, int]:
        """Calculate optimal batch sizes for dual RTX 4090"""
        config = cls.GEMMA_3N_CONFIGS.get(model_name, cls.GEMMA_3N_CONFIGS["gemma-2-9b"])
        model_size_gb = config["estimated_size_gb"]
        
        # Available memory per GPU after model
        available_per_gpu = (cls.VRAM_PER_GPU * cls.MEMORY_EFFICIENCY) - (model_size_gb / 2)
        
        # Estimate memory per sample (rough calculation)
        # Activations: hidden_size * sequence_length * 4 bytes * layers / (1024^3)
        activation_per_sample = (
            config["hidden_size"] * sequence_length * 4 * config["num_layers"] / (1024**3)
        )
        
        # Calculate batch sizes
        single_gpu_batch = max(1, int(available_per_gpu / activation_per_sample))
        total_batch = single_gpu_batch * 2  # 2 GPUs
        
        return {
            "per_device_batch_size": single_gpu_batch,
            "total_batch_size": total_batch,
            "gradient_accumulation_steps": max(1, 32 // total_batch),
            "effective_batch_size": total_batch * max(1, 32 // total_batch),
        }

class MockGemmaModel(nn.Module):
    """Mock Gemma model for testing distributed training"""
    
    def __init__(self, config_name: str = "gemma-2-9b"):
        super().__init__()
        self.config = RTX4090Config.GEMMA_3N_CONFIGS[config_name]
        
        # Create model layers similar to Gemma architecture
        self.embed_tokens = nn.Embedding(
            self.config["vocab_size"], 
            self.config["hidden_size"]
        )
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_layer(i) 
            for i in range(self.config["num_layers"])
        ])
        
        self.norm = nn.RMSNorm(self.config["hidden_size"])
        self.lm_head = nn.Linear(
            self.config["hidden_size"], 
            self.config["vocab_size"], 
            bias=False
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_transformer_layer(self, layer_idx: int) -> nn.Module:
        """Create a single transformer layer"""
        class TransformerLayer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(
                    config["hidden_size"],
                    config["num_attention_heads"],
                    batch_first=True
                )
                
                # MLP layers (target for LoRA)
                self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"])
                self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"])
                self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"])
                
                # Attention projections (target for LoRA)
                self.q_proj = nn.Linear(config["hidden_size"], config["hidden_size"])
                self.k_proj = nn.Linear(config["hidden_size"], config["hidden_size"])
                self.v_proj = nn.Linear(config["hidden_size"], config["hidden_size"])
                self.o_proj = nn.Linear(config["hidden_size"], config["hidden_size"])
                
                self.input_layernorm = nn.RMSNorm(config["hidden_size"])
                self.post_attention_layernorm = nn.RMSNorm(config["hidden_size"])
            
            def forward(self, x):
                # Self attention
                residual = x
                x = self.input_layernorm(x)
                
                # Apply projections (these will be replaced with LoRA)
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                
                attn_out, _ = self.self_attn(q, k, v)
                attn_out = self.o_proj(attn_out)
                x = residual + attn_out
                
                # MLP
                residual = x
                x = self.post_attention_layernorm(x)
                x = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
                x = residual + x
                
                return x
        
        return TransformerLayer(self.config)
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embed_tokens(input_ids)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class DistributedGemmaTest:
    """Main test class for distributed Gemma training"""
    
    def __init__(self, model_name: str = "gemma-2-9b"):
        self.model_name = model_name
        self.config = RTX4090Config.GEMMA_3N_CONFIGS[model_name]
        self.batch_config = RTX4090Config.get_optimal_batch_size(model_name)
        
        # Setup distributed environment
        self.setup_distributed()
    
    def setup_distributed(self):
        """Setup distributed training environment"""
        print("🚀 Setting up distributed environment for dual RTX 4090...")
        
        # Enable distributed training
        enable_distributed_training()
        
        # Print GPU information
        if torch.cuda.is_available():
            print(f"📊 CUDA Devices Available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        print(f"🌍 World Size: {get_world_size()}")
        print(f"🏷️  Rank: {get_rank()}")
        print(f"👑 Is Main Process: {is_main_process()}")
    
    def create_distributed_config(self, backend: str = "fsdp") -> DistributedConfig:
        """Create distributed configuration optimized for dual RTX 4090"""
        
        # Determine if we need CPU offloading based on model size
        model_size_gb = self.config["estimated_size_gb"]
        needs_cpu_offload = model_size_gb > (RTX4090Config.TOTAL_VRAM * 0.7)
        
        backend_enum = getattr(DistributedBackend, backend.upper(), DistributedBackend.FSDP)
        
        config = DistributedConfig(
            backend=backend_enum,
            world_size=2,  # Dual GPU setup
            mixed_precision=True,  # Use bfloat16 on RTX 4090
            gradient_checkpointing=True,  # Save memory
            cpu_offload=needs_cpu_offload,
            offload_optimizer_device="cpu" if needs_cpu_offload else "cuda",
            offload_param_device="cpu" if needs_cpu_offload else "cuda",
            bucket_cap_mb=50,  # Optimize for RTX 4090 memory bandwidth
            overlap_communication=True,
            find_unused_parameters=False,
            broadcast_buffers=True,
            gradient_accumulation_steps=self.batch_config["gradient_accumulation_steps"],
            distribute_lora=True,
            shard_parameters=True,
            sync_module_states=True,
            use_orig_params=False,  # Better memory efficiency
            activation_checkpointing=True,
        )
        
        return config
    
    def create_lora_config(self) -> DistributedLoRAConfig:
        """Create LoRA configuration optimized for Gemma"""
        return DistributedLoRAConfig(
            r=16,  # Good balance for Gemma
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"      # MLP layers
            ],
            lora_dropout=0.1,
            shard_strategy="hybrid",  # Optimal for dual GPU
            enable_gradient_checkpointing=True,
            use_rslora=False,  # Standard LoRA for initial testing
            use_dora=False,
        )
    
    def test_model_creation(self):
        """Test creating and wrapping model with distributed setup"""
        print(f"\n🧪 Testing model creation for {self.model_name}...")
        
        try:
            # Create base model
            model = MockGemmaModel(self.model_name)
            print(f"✅ Created base model with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Create configurations
            dist_config = self.create_distributed_config("fsdp")
            lora_config = self.create_lora_config()
            
            # Create distributed model
            distributed_model = UnslothDistributedModel(
                base_model=model,
                distributed_config=dist_config,
                lora_config=lora_config
            )
            
            print("✅ Successfully created distributed model")
            
            # Test parameter grouping
            param_groups = distributed_model.get_distributed_parameters()
            for group_name, params in param_groups.items():
                param_count = sum(p.numel() for p in params)
                print(f"  {group_name}: {param_count:,} parameters")
            
            return distributed_model
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_distributed_setup(self, backend: str = "fsdp"):
        """Test distributed environment setup"""
        print(f"\n🧪 Testing distributed setup with {backend.upper()}...")
        
        try:
            # Create configuration
            dist_config = self.create_distributed_config(backend)
            
            # Setup environment
            env_setup = DistributedEnvironmentSetup(dist_config)
            
            # Test accelerator setup if using accelerate
            if backend in ["ddp", "fsdp"]:
                accelerator = env_setup.setup_accelerator()
                if accelerator:
                    print("✅ Accelerator setup successful")
                else:
                    print("⚠️ Accelerator setup skipped (accelerate not available)")
            
            print(f"✅ Distributed setup completed for {backend.upper()}")
            return env_setup
            
        except Exception as e:
            print(f"❌ Distributed setup failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_memory_management(self):
        """Test memory management and optimization"""
        print(f"\n🧪 Testing memory management for {self.model_name}...")
        
        try:
            # Create device manager
            device_manager = DeviceManager()
            
            # Check memory availability
            memory_info = device_manager.get_memory_info()
            print("📊 Memory Information:")
            for i, info in enumerate(memory_info):
                print(f"  GPU {i}: {info['allocated']:.1f}GB / {info['total']:.1f}GB "
                      f"({info['utilization']:.1%} utilized)")
            
            # Test optimal batch size calculation
            print(f"\n📈 Optimal Batch Configuration:")
            for key, value in self.batch_config.items():
                print(f"  {key}: {value}")
            
            # Calculate memory requirements
            device_manager.calculate_memory_requirements(
                model_size_gb=self.config["estimated_size_gb"],
                batch_size=self.batch_config["per_device_batch_size"],
                sequence_length=2048,
                num_devices=2
            )
            
            print("✅ Memory management test completed")
            return True
            
        except Exception as e:
            print(f"❌ Memory management test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_forward_pass(self, distributed_model):
        """Test forward pass with distributed model"""
        print(f"\n🧪 Testing forward pass...")
        
        try:
            # Create dummy input
            batch_size = self.batch_config["per_device_batch_size"]
            seq_length = 512  # Shorter sequence for testing
            vocab_size = self.config["vocab_size"]
            
            # Generate random input
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            labels = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            # Move to GPU if available
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                distributed_model = distributed_model.cuda()
            
            print(f"📝 Input shape: {input_ids.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = distributed_model(input_ids=input_ids, labels=labels)
            
            if "logits" in outputs:
                print(f"✅ Forward pass successful - Logits shape: {outputs['logits'].shape}")
            
            if "loss" in outputs and outputs["loss"] is not None:
                print(f"✅ Loss calculation successful - Loss: {outputs['loss'].item():.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_training_step(self, distributed_model):
        """Test a single training step"""
        print(f"\n🧪 Testing training step...")
        
        try:
            # Create optimizer
            optimizer = torch.optim.AdamW(
                distributed_model.parameters(), 
                lr=2e-5, 
                weight_decay=0.01
            )
            
            # Create dummy input
            batch_size = self.batch_config["per_device_batch_size"]
            seq_length = 512
            vocab_size = self.config["vocab_size"]
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            labels = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                distributed_model = distributed_model.cuda()
            
            # Set model to training mode
            distributed_model.train()
            
            # Forward pass
            outputs = distributed_model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            
            print(f"📊 Initial loss: {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Sync distributed parameters
            distributed_model.sync_distributed_parameters()
            
            print("✅ Training step completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🚀 Starting Comprehensive Distributed Training Test")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Target Hardware: Dual RTX 4090 (2x 24GB)")
        print(f"Estimated Model Size: {self.config['estimated_size_gb']:.1f} GB")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Memory Management
        results["memory_management"] = self.test_memory_management()
        
        # Test 2: Model Creation
        distributed_model = self.test_model_creation()
        results["model_creation"] = distributed_model is not None
        
        if not distributed_model:
            print("❌ Stopping tests - model creation failed")
            return results
        
        # Test 3: Distributed Setup (test multiple backends)
        backends_to_test = ["fsdp", "ddp"]
        for backend in backends_to_test:
            setup_result = self.test_distributed_setup(backend)
            results[f"setup_{backend}"] = setup_result is not None
        
        # Test 4: Forward Pass
        results["forward_pass"] = self.test_forward_pass(distributed_model)
        
        # Test 5: Training Step
        results["training_step"] = self.test_training_step(distributed_model)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if passed_test:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Ready for dual RTX 4090 training.")
        else:
            print("⚠️ Some tests failed. Check logs above for details.")
        
        return results

def main():
    """Main test function"""
    print("🚀 Unsloth Distributed Training Test Suite")
    print("Optimized for Gemma 3N on Dual RTX 4090 Setup")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        ("gemma-2-9b", "Small model test (should fit easily)"),
        ("gemma-2-27b", "Large model test (requires CPU offloading)"),
    ]
    
    all_results = {}
    
    for model_name, description in test_configs:
        print(f"\n🧪 Testing {model_name}: {description}")
        print("-" * 50)
        
        try:
            tester = DistributedGemmaTest(model_name)
            results = tester.run_comprehensive_test()
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ Test suite failed for {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Final summary
    print("\n" + "🎯 FINAL SUMMARY" + "=" * 45)
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name}: ❌ ERROR - {results['error']}")
        else:
            passed = sum(1 for r in results.values() if r)
            total = len(results)
            print(f"{model_name}: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    print("\n✨ Test suite completed!")

if __name__ == "__main__":
    main()
