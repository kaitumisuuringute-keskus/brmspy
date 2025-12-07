#!/bin/bash
# brmspy Virtual Environment Setup Script
# Uses uv (fast Python package installer) for environment management

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
PYTHON_VERSION="3.12"  # Default Python version (minimum: 3.8, recommended: 3.12)

echo -e "${BLUE}=== brmspy Virtual Environment Setup ===${NC}"
echo ""

# Function to print colored messages
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
info "Checking for uv installation..."
if ! command -v uv &> /dev/null; then
    warning "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify installation
    if command -v uv &> /dev/null; then
        success "uv installed successfully!"
    else
        error "Failed to install uv. Please install manually from https://github.com/astral-sh/uv"
        exit 1
    fi
else
    success "uv is already installed ($(uv --version))"
fi

# Check Python version
info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

CURRENT_PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
info "Found Python $CURRENT_PYTHON_VERSION"

# Parse version numbers
MAJOR=$(echo $CURRENT_PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $CURRENT_PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    error "Python 3.8 or higher is required. Found: $CURRENT_PYTHON_VERSION"
    error "Please upgrade Python: https://www.python.org/downloads/"
    exit 1
fi

success "Python version is compatible!"

# Remove existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    warning "Existing virtual environment found at $VENV_DIR"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        success "Removed existing environment"
    else
        info "Keeping existing environment and updating packages..."
    fi
fi

# Create virtual environment with uv
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python "$PYTHON_CMD"
    success "Virtual environment created at $VENV_DIR"
fi

# Activate virtual environment
info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip, setuptools, wheel using uv
info "Upgrading core packages..."
uv pip install --upgrade pip setuptools wheel

# Install package in development mode with all dependencies
info "Installing brmspy in development mode..."
info "This will install:"
info "  - Core dependencies (cmdstanpy, numpy, pandas, rpy2)"
info "  - Development tools (pytest, black, ruff, mypy)"
info "  - Documentation tools (nbdev, jupyter, sphinx)"
info "  - Visualization tools (arviz, matplotlib, seaborn)"
echo ""

# Install with all optional dependencies
uv pip install -e ".[all]"
uv pip install mkdocs-material mkdocs-jupyter "mkdocstrings[python]"

success "All packages installed successfully!"

# Display installed packages
echo ""
info "Installed packages:"
uv pip list | head -20
echo "  ... (showing first 20)"
echo ""

# Check if CmdStan needs to be installed
info "Checking CmdStan installation..."
python -c "
import cmdstanpy
import os
try:
    cmdstan_path = cmdstanpy.cmdstan_path()
    print(f'CmdStan found at: {cmdstan_path}')
except:
    print('CmdStan not yet installed. It will be installed automatically on first use.')
    print('You can also install it manually with:')
    print('  python -c \"import cmdstanpy; cmdstanpy.install_cmdstan()\"')
" || true

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Virtual environment is activated. To use it:"
echo ""
echo -e "${YELLOW}  # Already activated in this shell${NC}"
echo -e "${YELLOW}  # For new shells, run:${NC}"
echo -e "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Install R and brms: python -c 'import brmspy; brmspy.install_brms()'"
echo "  2. Run tests: pytest"
echo "  3. Start development: jupyter notebook"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""

# Create activation helper script
cat > activate.sh << 'EOF'
#!/bin/bash
# Quick activation script for brmspy development environment
source .venv/bin/activate
echo "brmspy development environment activated!"
echo "Python: $(python --version)"
echo "Location: $(which python)"
EOF

chmod +x activate.sh
info "Created activate.sh helper script for quick activation"

echo ""
success "Setup complete! Happy coding! 🎉"
