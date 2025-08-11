#!/bin/bash

# GAS Installation Script (using standard venv)
# Installs the GAS package with CLI tool using standard venv

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════╗"
echo "║       ⛽️  GAS Installation ⛽️             ║"
echo "║   Installing gas package and gascli       ║"
echo "╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    log_error "pyproject.toml not found. Please run this script from the project root directory."
    exit 1
fi

# Check Python version
log_info "Checking Python version..."
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "")
if [ -z "$python_version" ]; then
    log_error "Python 3 is not installed or not in PATH."
    exit 1
fi

major=$(echo $python_version | cut -d'.' -f1)
minor=$(echo $python_version | cut -d'.' -f2)

if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 10 ]); then
    log_error "Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi

# Python 3.10+ is supported with no upper version limit

log_success "Python $python_version detected ✓"

# Check and install required system dependencies
log_info "Checking system dependencies..."
if ! command -v pkg-config >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    log_info "Installing required system dependencies..."
    if command -v apt-get >/dev/null 2>&1; then
        # For Debian/Ubuntu systems
        if ! apt-get update && apt-get install -y pkg-config cmake ffmpeg; then
            log_error "Failed to install system dependencies via apt-get"
            log_error "Please install pkg-config, cmake, and ffmpeg manually"
            exit 1
        fi
    else
        log_error "Could not find apt-get. Please install pkg-config, cmake, and ffmpeg manually"
        log_error "These are required for building some Python packages"
        exit 1
    fi
    log_success "System dependencies installed ✓"
else
    log_success "System dependencies already installed ✓"
fi

# Check and install Node.js and npm
log_info "Checking for Node.js and npm..."
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    log_info "Installing Node.js and npm..."
    if command -v apt-get >/dev/null 2>&1; then
        # For Debian/Ubuntu systems
        if ! apt-get update && apt-get install -y nodejs npm; then
            log_error "Failed to install Node.js and npm via apt-get"
            log_error "Please install Node.js and npm manually"
            exit 1
        fi
    else
        log_error "Could not find apt-get. Please install Node.js and npm manually"
        log_error "These are required for PM2 installation"
        exit 1
    fi
    log_success "Node.js and npm installed ✓"
else
    log_success "Node.js and npm already installed ✓"
fi

# Install PM2 globally if not already installed
log_info "Checking for PM2..."
if ! command -v pm2 >/dev/null 2>&1; then
    log_info "Installing PM2 globally..."
    if ! npm install -g pm2; then
        log_error "Failed to install PM2 globally"
        log_error "Please install PM2 manually: npm install -g pm2"
        exit 1
    fi
    log_success "PM2 installed globally ✓"
else
    log_success "PM2 already installed ✓"
fi

# Check if uv is installed
log_info "Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
    log_error "uv is not installed. Please install uv first:"
    log_error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    log_error "  or"
    log_error "  pip install uv"
    exit 1
fi

log_success "uv detected ✓"

# Remove existing virtual environment to ensure fresh install
if [ -d ".venv" ]; then
    log_info "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment and install dependencies with uv
log_info "Creating virtual environment and installing dependencies with uv..."
uv sync

# Install the gas package (this creates the gascli entry point)
log_info "Installing gas package..."
uv pip install -e .

# Install additional git dependencies
log_info "Installing additional git dependencies..."
source .venv/bin/activate && uv pip install git+https://github.com/deepseek-ai/Janus.git
log_success "Git dependencies installed ✓"

log_success "Virtual environment created and dependencies installed ✓"

# Verify installation
log_info "Verifying installation..."

# Check if gascli command is available in the virtual environment
if .venv/bin/gascli --help >/dev/null 2>&1; then
    log_success "gascli command installed successfully ✓"
else
    log_error "gascli command not found in virtual environment."
    log_warning "Checking if package was installed correctly..."
    
    # Check if the package is installed
    if pip show gas >/dev/null 2>&1; then
        log_success "gas package is installed ✓"
    else
        log_error "gas package is not installed."
    fi
fi

# Test basic CLI functionality
log_info "Testing CLI functionality..."
if .venv/bin/gascli --help >/dev/null 2>&1; then
    log_success "CLI help system working ✓"
else
    log_warning "CLI help command failed. Installation may be incomplete."
fi

echo
log_success "🎉 GAS installation completed!"
echo
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}1. Activate Virtual Environment:${NC}"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo -e "  or, less conveniently, run gascli with ${YELLOW}.venv/bin/gascli${NC}"
echo
echo -e "${GREEN}2. CLI Quick Start:${NC}"
echo -e "  ${YELLOW}gascli --help${NC}                   Show main help"
echo -e "  ${YELLOW}gascli validator start${NC}          Start validator services"
echo -e "  ${YELLOW}gascli miner push-discriminator${NC} Push a model"
echo
echo -e "  ${GREEN}Available Aliases:${NC}"
echo -e "    ${YELLOW}validator${NC} → ${YELLOW}vali${NC}, ${YELLOW}v${NC}"
echo -e "    ${YELLOW}miner${NC} → ${YELLOW}m${NC}"
echo
echo -e "${GREEN}3. Important Notes:${NC}"
echo -e "  ${YELLOW}Validators:${NC} Make sure to update your .env.validator file before starting"
echo
echo -e "${BLUE}═══════════════════════════════════════════${NC}" 