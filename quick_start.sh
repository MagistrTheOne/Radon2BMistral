#!/bin/bash

# RADON Mistral Quick Start Script
# This script sets up and runs the RADON Mistral framework

set -e

echo "🚀 RADON Mistral Quick Start"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.9 or higher."
        exit 1
    fi
}

# Check if CUDA is available
check_cuda() {
    print_status "Checking CUDA availability..."
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | cut -d' ' -f9)
        print_success "CUDA $CUDA_VERSION found"
        DEVICE="cuda"
    else
        print_warning "CUDA not found. Using CPU mode."
        DEVICE="cpu"
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Try to install Flash Attention (optional)
    print_status "Installing Flash Attention (optional)..."
    pip install flash-attn --no-build-isolation || print_warning "Flash Attention installation failed, continuing without it"
    
    print_success "Dependencies installed"
}

# Create necessary directories
create_dirs() {
    print_status "Creating necessary directories..."
    mkdir -p logs outputs checkpoints models/checkpoint tokenizer/checkpoint data
    print_success "Directories created"
}

# Create sample data
create_sample_data() {
    print_status "Creating sample training data..."
    
    # Create sample Russian corpus
    cat > data/sample_corpus_ru.txt << EOF
Привет! Это примерный корпус для обучения русского токенизатора.
Машинное обучение и обработка естественного языка - важные области.
Глубокое обучение совершило революцию в ИИ.
Трансформеры изменили подход к обработке текста.
Русский язык имеет богатую морфологию.
EOF

    # Create sample English corpus
    cat > data/sample_corpus_en.txt << EOF
Hello! This is a sample corpus for training the English tokenizer.
Machine learning and natural language processing are important fields.
Deep learning has revolutionized AI.
Transformers have changed the approach to text processing.
English is a widely spoken language.
EOF

    # Create combined training data
    cat > data/training_data.txt << EOF
Привет! Это примерный корпус для обучения русского токенизатора.
Hello! This is a sample corpus for training the English tokenizer.
Машинное обучение и обработка естественного языка - важные области.
Machine learning and natural language processing are important fields.
Глубокое обучение совершило революцию в ИИ.
Deep learning has revolutionized AI.
Трансформеры изменили подход к обработке текста.
Transformers have changed the approach to text processing.
def hello_world():
    print("Hello, World!")

import torch
import torch.nn as nn
EOF

    print_success "Sample data created"
}

# Train tokenizer
train_tokenizer() {
    print_status "Training hybrid tokenizer..."
    
    if [ ! -d "tokenizer_output" ]; then
        python tokenizer/train_hybrid_tokenizer.py \
            --input_ru data/sample_corpus_ru.txt \
            --input_en data/sample_corpus_en.txt \
            --output_dir ./tokenizer_output \
            --vocab_size_ru 16000 \
            --vocab_size_en 16000
        
        print_success "Hybrid tokenizer trained"
    else
        print_status "Tokenizer already exists, skipping training"
    fi
}

# Train model (optional)
train_model() {
    if [ "$1" = "--train-model" ]; then
        print_status "Training Mistral model..."
        
        python scripts/train_mistral.py \
            --model_config configs/model_config_mistral_2b.json \
            --data_path data/training_data.txt \
            --tokenizer_path ./tokenizer_output \
            --output_dir ./outputs \
            --num_epochs 1 \
            --batch_size 4 \
            --device $DEVICE \
            --use_fp16 \
            --gradient_checkpointing
        
        print_success "Model training completed"
    else
        print_status "Skipping model training (use --train-model to enable)"
    fi
}

# Run demo
run_demo() {
    print_status "Running Mistral demo..."
    
    python scripts/demo_mistral.py \
        --config configs/model_config_mistral_2b.json \
        --tokenizer ./tokenizer_output \
        --device $DEVICE
    
    print_success "Demo completed"
}

# Start API server
start_api() {
    print_status "Starting API server..."
    
    # Set environment variables
    export MODEL_CONFIG_PATH=configs/model_config_mistral_2b.json
    export MODEL_PATH=./models/checkpoint
    export TOKENIZER_PATH=./tokenizer_output
    export DEVICE=$DEVICE
    export USE_FLASH_ATTENTION=true
    
    # Start server in background
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Test API
    print_status "Testing API..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API server is running on http://localhost:8000"
        print_status "API documentation: http://localhost:8000/docs"
        print_status "Health check: http://localhost:8000/health"
        
        # Test generation
        print_status "Testing text generation..."
        curl -X POST "http://localhost:8000/api/v1/generate" \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "Привет, RADON!",
                "max_length": 100,
                "temperature": 0.7
            }' | python -m json.tool
        
        print_success "API test completed"
    else
        print_error "API server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
}

# Main function
main() {
    echo "Starting RADON Mistral setup..."
    echo
    
    # Check prerequisites
    check_python
    check_cuda
    echo
    
    # Setup environment
    setup_venv
    install_deps
    create_dirs
    create_sample_data
    echo
    
    # Train components
    train_tokenizer
    train_model "$1"
    echo
    
    # Run demo
    run_demo
    echo
    
    # Start API
    start_api
    echo
    
    print_success "RADON Mistral setup completed successfully!"
    echo
    echo "🎉 What's next?"
    echo "==============="
    echo "1. API Server: http://localhost:8000"
    echo "2. API Docs: http://localhost:8000/docs"
    echo "3. Health Check: http://localhost:8000/health"
    echo "4. Test Generation: curl -X POST http://localhost:8000/api/v1/generate"
    echo
    echo "📚 Documentation:"
    echo "- README.md: Basic usage"
    echo "- ARCHITECTURE.md: Technical details"
    echo "- MIGRATION.md: Migration guide"
    echo
    echo "🔧 Commands:"
    echo "- Stop server: kill $SERVER_PID"
    echo "- Run demo: python scripts/demo_mistral.py"
    echo "- Run benchmark: python scripts/benchmark_mistral.py"
    echo "- Train model: python scripts/train_mistral.py"
    echo
    echo "Happy coding with RADON Mistral! 🚀"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "RADON Mistral Quick Start Script"
        echo
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --train-model    Train the model after setup"
        echo "  --help, -h       Show this help message"
        echo
        echo "Examples:"
        echo "  $0                # Basic setup without training"
        echo "  $0 --train-model  # Setup with model training"
        echo
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac
