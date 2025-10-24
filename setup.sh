#!/bin/bash
# Neuromorphic Continual Learning System Setup Script
# Run with: bash setup.sh

set -e  # Exit on any error

echo "=========================================="
echo "Neuromorphic Continual Learning Setup"
echo "=========================================="

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

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_success "Python $python_version found (>= $required_version required)"
else
    print_error "Python >= $required_version required. Found: $python_version"
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits | head -1)
    print_success "CUDA GPU detected: $gpu_info"
    USE_GPU=true
else
    print_warning "No CUDA GPU detected. Will install CPU-only versions."
    USE_GPU=false
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch based on CUDA availability
if [ "$USE_GPU" = true ]; then
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_status "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install the package in development mode
print_status "Installing neuromorphic continual learning system..."
pip install -e .

# Install optional dependencies based on GPU availability
if [ "$USE_GPU" = true ]; then
    print_status "Installing GPU-specific dependencies..."
    pip install faiss-gpu
else
    print_status "Installing CPU-specific dependencies..."
    pip install faiss-cpu
fi

# Install development dependencies
read -p "Install development dependencies (for contributing)? [y/N]: " install_dev
if [[ $install_dev =~ ^[Yy]$ ]]; then
    print_status "Installing development dependencies..."
    pip install -e ".[dev]"
    
    # Setup pre-commit hooks
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs checkpoints outputs data
print_success "Directories created"

# Generate default configuration
print_status "Generating default configuration..."
if [ ! -f "config.yaml" ]; then
    cp configs/default_config.yaml config.yaml
    print_success "Default configuration created: config.yaml"
else
    print_warning "config.yaml already exists, skipping"
fi

# Test installation
print_status "Testing installation..."
if python -c "import neuromorphic_cl; print('Import successful')" 2>/dev/null; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Run basic demo if requested
read -p "Run basic demo to test the system? [y/N]: " run_demo
if [[ $run_demo =~ ^[Yy]$ ]]; then
    print_status "Running basic demo..."
    python examples/basic_usage.py
fi

echo ""
echo "=========================================="
print_success "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Edit config.yaml to point to your datasets"
echo "3. Start training: neuromorphic-train train --config config.yaml"
echo ""
echo "For medical imaging tasks, use:"
echo "cp configs/medical_config.yaml config.yaml"
echo ""
echo "Available commands:"
echo "- neuromorphic-train: Training and configuration"
echo "- neuromorphic-eval: Evaluation and analysis"  
echo "- neuromorphic-infer: Inference on new data"
echo ""
echo "Documentation: README.md"
echo "Examples: examples/"
echo "Configurations: configs/"

# Deactivate virtual environment
deactivate
