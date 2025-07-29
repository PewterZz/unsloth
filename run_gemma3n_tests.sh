#!/bin/bash

# Test runner for Gemma 3N distributed training on dual RTX 4090
# This script tests various configurations and backends

set -e  # Exit on any error

echo "🚀 Unsloth Distributed Training Test Runner"
echo "Target: Gemma 3N on Dual RTX 4090 Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU setup
check_gpu_setup() {
    print_status $BLUE "🔍 Checking GPU Setup..."
    
    if ! command_exists nvidia-smi; then
        print_status $RED "❌ nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    # Check number of GPUs
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_status $GREEN "📊 Found $GPU_COUNT GPU(s)"
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        print_status $YELLOW "⚠️  Warning: Less than 2 GPUs detected. Some tests may be skipped."
    fi
    
    # Show GPU information
    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | nl -v0 -s": " | while read line; do
        echo "  GPU $line"
    done
    echo ""
}

# Function to check Python environment
check_python_env() {
    print_status $BLUE "🐍 Checking Python Environment..."
    
    # Check Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status $GREEN "Python version: $PYTHON_VERSION"
    
    if [ "$(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.8" ]; then
        print_status $RED "❌ Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    # Check required packages
    REQUIRED_PACKAGES=("torch" "transformers" "accelerate" "peft")
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            print_status $GREEN "✅ $package: $VERSION"
        else
            print_status $YELLOW "⚠️  $package not found (will test fallback behavior)"
        fi
    done
    
    # Check optional packages
    OPTIONAL_PACKAGES=("deepspeed" "flash_attn")
    
    for package in "${OPTIONAL_PACKAGES[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            print_status $GREEN "✅ $package (optional): $VERSION"
        else
            print_status $YELLOW "⚠️  $package (optional) not found"
        fi
    done
    
    echo ""
}

# Function to run single GPU test
run_single_gpu_test() {
    print_status $BLUE "🧪 Running Single GPU Test..."
    
    CUDA_VISIBLE_DEVICES=0 python test_gemma3n_dual_4090.py 2>&1 | tee logs/single_gpu_test.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_status $GREEN "✅ Single GPU test passed"
        return 0
    else
        print_status $RED "❌ Single GPU test failed"
        return 1
    fi
}

# Function to run dual GPU test with torchrun
run_dual_gpu_torchrun() {
    local backend=${1:-fsdp}
    print_status $BLUE "🧪 Running Dual GPU Test with torchrun (backend: $backend)..."
    
    # Set environment variables for the backend
    export DISTRIBUTED_BACKEND=$backend
    
    torchrun --nproc_per_node=2 --nnodes=1 test_gemma3n_dual_4090.py 2>&1 | tee logs/dual_gpu_${backend}_torchrun.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_status $GREEN "✅ Dual GPU torchrun test passed (backend: $backend)"
        return 0
    else
        print_status $RED "❌ Dual GPU torchrun test failed (backend: $backend)"
        return 1
    fi
}

# Function to run dual GPU test with accelerate
run_dual_gpu_accelerate() {
    print_status $BLUE "🧪 Running Dual GPU Test with accelerate..."
    
    if ! command_exists accelerate; then
        print_status $YELLOW "⚠️  accelerate not found, skipping test"
        return 0
    fi
    
    # Create accelerate config
    cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    
    accelerate launch --config_file accelerate_config.yaml test_gemma3n_dual_4090.py 2>&1 | tee logs/dual_gpu_accelerate.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_status $GREEN "✅ Dual GPU accelerate test passed"
        return 0
    else
        print_status $RED "❌ Dual GPU accelerate test failed"
        return 1
    fi
}

# Function to run DeepSpeed test
run_deepspeed_test() {
    print_status $BLUE "🧪 Running DeepSpeed Test..."
    
    if ! command_exists deepspeed; then
        print_status $YELLOW "⚠️  DeepSpeed not found, skipping test"
        return 0
    fi
    
    # Create DeepSpeed config
    cat > deepspeed_config.json << EOF
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 50000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 50000000,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 4
    },
    "wall_clock_breakdown": false
}
EOF
    
    deepspeed --num_gpus=2 test_gemma3n_dual_4090.py --deepspeed deepspeed_config.json 2>&1 | tee logs/deepspeed_test.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_status $GREEN "✅ DeepSpeed test passed"
        return 0
    else
        print_status $RED "❌ DeepSpeed test failed"
        return 1
    fi
}

# Function to run memory benchmark
run_memory_benchmark() {
    print_status $BLUE "🧪 Running Memory Benchmark..."
    
    python -c "
import torch
from test_gemma3n_dual_4090 import RTX4090Config, DistributedGemmaTest

print('📊 Memory Benchmark for Dual RTX 4090')
print('=' * 50)

# Test different model sizes
models = ['gemma-2-9b', 'gemma-2-27b'] 

for model_name in models:
    print(f'\\n🧪 Testing {model_name}:')
    config = RTX4090Config.GEMMA_3N_CONFIGS[model_name]
    batch_config = RTX4090Config.get_optimal_batch_size(model_name)
    
    print(f'  Estimated model size: {config[\"estimated_size_gb\"]:.1f} GB')
    print(f'  Optimal batch size per GPU: {batch_config[\"per_device_batch_size\"]}')
    print(f'  Total effective batch size: {batch_config[\"effective_batch_size\"]}')
    print(f'  Gradient accumulation steps: {batch_config[\"gradient_accumulation_steps\"]}')
    
    # Check if model fits in memory
    total_vram = RTX4090Config.TOTAL_VRAM * RTX4090Config.MEMORY_EFFICIENCY
    if config['estimated_size_gb'] <= total_vram:
        print(f'  ✅ Model fits in GPU memory ({config[\"estimated_size_gb\"]:.1f}GB <= {total_vram:.1f}GB)')
    else:
        print(f'  ⚠️  Model requires CPU offloading ({config[\"estimated_size_gb\"]:.1f}GB > {total_vram:.1f}GB)')

print('\\n✨ Memory benchmark completed!')
" 2>&1 | tee logs/memory_benchmark.log
}

# Function to create summary report
create_summary_report() {
    print_status $BLUE "📋 Creating Summary Report..."
    
    REPORT_FILE="test_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# Unsloth Distributed Training Test Report

**Date**: $(date)
**Target Hardware**: Dual RTX 4090 (2x 24GB VRAM)
**Target Model**: Gemma 3N

## System Information

### GPU Configuration
\`\`\`
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl -v0 -s": GPU ")
\`\`\`

### Python Environment
- Python Version: $(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
- PyTorch Version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not available")

## Test Results

EOF
    
    # Add test results to report
    for log_file in logs/*.log; do
        if [ -f "$log_file" ]; then
            test_name=$(basename "$log_file" .log)
            echo "### $test_name" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
            tail -20 "$log_file" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
        fi
    done
    
    print_status $GREEN "📄 Report saved to: $REPORT_FILE"
}

# Main execution
main() {
    # Parse command line arguments
    TEST_TYPE=${1:-all}
    
    # Create logs directory
    mkdir -p logs
    
    # Clean up old logs
    rm -f logs/*.log
    
    print_status $BLUE "Starting test suite with type: $TEST_TYPE"
    
    # System checks
    check_gpu_setup
    check_python_env
    
    # Initialize test results
    PASSED_TESTS=0
    TOTAL_TESTS=0
    
    # Run tests based on type
    case $TEST_TYPE in
        "single")
            print_status $BLUE "Running single GPU tests only..."
            run_single_gpu_test && ((PASSED_TESTS++))
            ((TOTAL_TESTS++))
            ;;
        "dual")
            print_status $BLUE "Running dual GPU tests only..."
            run_dual_gpu_torchrun fsdp && ((PASSED_TESTS++))
            ((TOTAL_TESTS++))
            ;;
        "memory")
            print_status $BLUE "Running memory benchmark only..."
            run_memory_benchmark && ((PASSED_TESTS++))
            ((TOTAL_TESTS++))
            ;;
        "all"|*)
            print_status $BLUE "Running comprehensive test suite..."
            
            # Memory benchmark
            run_memory_benchmark && ((PASSED_TESTS++))
            ((TOTAL_TESTS++))
            
            # Single GPU test
            run_single_gpu_test && ((PASSED_TESTS++))
            ((TOTAL_TESTS++))
            
            # Dual GPU tests with different methods
            if [ "$GPU_COUNT" -ge 2 ]; then
                # Test different backends
                for backend in fsdp ddp; do
                    run_dual_gpu_torchrun $backend && ((PASSED_TESTS++))
                    ((TOTAL_TESTS++))
                done
                
                # Test with accelerate
                run_dual_gpu_accelerate && ((PASSED_TESTS++))
                ((TOTAL_TESTS++))
                
                # Test with DeepSpeed
                run_deepspeed_test && ((PASSED_TESTS++))
                ((TOTAL_TESTS++))
            else
                print_status $YELLOW "⚠️  Skipping dual GPU tests (insufficient GPUs)"
            fi
            ;;
    esac
    
    # Final results
    echo ""
    print_status $BLUE "🎯 TEST SUMMARY"
    echo "=================================================="
    print_status $GREEN "Passed: $PASSED_TESTS/$TOTAL_TESTS tests"
    
    if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
        print_status $GREEN "🎉 All tests passed! Your dual RTX 4090 setup is ready for Gemma 3N distributed training."
    else
        print_status $YELLOW "⚠️  Some tests failed. Check the logs for details."
    fi
    
    # Create report
    create_summary_report
    
    # Cleanup temporary files
    rm -f accelerate_config.yaml deepspeed_config.json
    
    print_status $BLUE "✨ Test suite completed!"
}

# Help function
show_help() {
    echo "Usage: $0 [TEST_TYPE]"
    echo ""
    echo "TEST_TYPE options:"
    echo "  all     - Run all tests (default)"
    echo "  single  - Run single GPU tests only"
    echo "  dual    - Run dual GPU tests only"
    echo "  memory  - Run memory benchmark only"
    echo ""
    echo "Examples:"
    echo "  $0              # Run all tests"
    echo "  $0 single       # Test single GPU setup"
    echo "  $0 dual         # Test dual GPU setup"
    echo "  $0 memory       # Run memory benchmark"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
