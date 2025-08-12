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

# Default values
SKIP_SYSTEM_DEPS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-system-deps)
            SKIP_SYSTEM_DEPS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-system-deps    Skip installation of system dependencies"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                 Install with all system dependencies"
            echo "  $0 --no-system-deps Install without system dependencies"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# Helper functions
check_command() {
    command -v "$1" >/dev/null 2>&1
}

check_apt_get() {
    if ! check_command apt-get; then
        log_error "Could not find apt-get. Please install $1 manually"
        exit 1
    fi
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

log_info "Checking for uv..."
if ! check_command uv; then
    log_error "uv is not installed. Please install uv first:"
    log_error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    log_error "  or"
    log_error "  pip install uv"
    exit 1
fi

log_success "uv detected ✓"

# Python 3.10+ is supported with no upper version limit

log_success "Python $python_version detected ✓"

# Check and install required system dependencies (unless --no-system-deps is specified)
if [ "$SKIP_SYSTEM_DEPS" = false ]; then
    log_info "Checking system dependencies..."
    if ! check_command pkg-config || ! check_command cmake || ! check_command ffmpeg; then
        log_info "Installing required system dependencies..."
        check_apt_get "pkg-config, cmake, and ffmpeg"
        
        if ! apt-get update && apt-get install -y pkg-config cmake ffmpeg; then
            log_error "Failed to install system dependencies via apt-get"
            log_error "Please install pkg-config, cmake, and ffmpeg manually"
            exit 1
        fi
        log_success "System dependencies installed ✓"
    else
        log_success "System dependencies already installed ✓"
    fi

    log_info "Installing ChromeDriver dependencies..."
    check_apt_get "Chrome, libnss3, libnspr4, and xvfb"
    
    # Install Chrome if not present
    if ! check_command google-chrome; then
        apt-get update && apt-get install -y wget gnupg2
        wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
        echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
        apt-get update && apt-get install -y google-chrome-stable
        log_success "Google Chrome installed ✓"
    else
        log_success "Google Chrome already installed ✓"
    fi
    
    # Install ChromeDriver dependencies and Xvfb
    apt-get update && apt-get install -y libnss3 libnspr4 xvfb
    log_success "ChromeDriver dependencies and Xvfb installed ✓"

    log_info "Checking for Node.js and npm..."
    if ! check_command node || ! check_command npm; then
        log_info "Installing Node.js and npm..."
        check_apt_get "Node.js and npm"
        
        if ! apt-get update && apt-get install -y nodejs npm; then
            log_error "Failed to install Node.js and npm via apt-get"
            log_error "Please install Node.js and npm manually"
            exit 1
        fi
        log_success "Node.js and npm installed ✓"
        
        # Refresh PATH to ensure npm is available
        export PATH="$PATH:/usr/local/bin:/usr/bin"
        hash -r  # Clear command hash table
        
        # Small delay to ensure system registers new binaries
        sleep 2
    else
        log_success "Node.js and npm already installed ✓"
    fi

    # Verify npm is available and working
    log_info "Verifying npm installation..."
    if ! npm --version >/dev/null 2>&1; then
        log_error "npm is not available or not working properly"
        exit 1
    fi
    
    log_success "npm verified and working ✓"

    log_info "Checking for PM2..."
    if ! check_command pm2; then
        log_info "Installing PM2 globally..."
        # Ensure we're using the correct npm
        npm_path=$(which npm)
        log_info "Using npm at: $npm_path"
        
        if ! npm install -g pm2; then
            log_error "Failed to install PM2 globally"
            log_error "Please install PM2 manually: npm install -g pm2"
            exit 1
        fi
        
        # Verify PM2 installation
        if ! check_command pm2; then
            log_error "PM2 was not found in PATH after installation"
            log_error "You may need to restart your shell or add npm global bin to PATH"
            exit 1
        fi
        
        log_success "PM2 installed globally ✓"
    else
        log_success "PM2 already installed ✓"
    fi

    log_info "Installing dotenv for ecosystem config..."
    if ! npm install dotenv; then
        log_error "Failed to install dotenv"
        log_error "Please install dotenv manually: npm install dotenv"
        exit 1
    fi
    log_success "dotenv installed ✓"
else
    log_warning "Skipping system dependencies installation (--no-system-deps specified)"
    log_warning "Make sure you have the following installed manually:"
    log_warning "  - pkg-config, cmake, ffmpeg"
    log_warning "  - Google Chrome, libnss3, libnspr4, xvfb"
    log_warning "  - Node.js, npm, PM2, dotenv"
fi


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

# Check if gascli command is available and working in the virtual environment
if .venv/bin/gascli --help >/dev/null 2>&1; then
    log_success "gascli command installed and working ✓"
else
    log_error "gascli command not found or not working in virtual environment."
    log_warning "Checking if package was installed correctly..."
    
    # Check if the package is installed
    if pip show gas >/dev/null 2>&1; then
        log_success "gas package is installed ✓"
        log_warning "gascli command may need to be reinstalled or virtual environment may need to be recreated."
    else
        log_error "gas package is not installed."
    fi
fi

echo
log_success "🎉 GAS installation completed!"
echo 
echo -e "${BLUE}═══════════════════ QUICKSTART ════════════════════${NC}"
echo
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
echo -e "  ${YELLOW}Validators:${NC} Make sure to update your .env.validator"
if [ "$SKIP_SYSTEM_DEPS" = true ]; then
    echo -e "  ${YELLOW}System Dependencies:${NC} You skipped system dependency installation"
    echo -e "  ${YELLOW}  Validators: ${NC} Make sure you have: pkg-config, cmake, ffmpeg, Chrome, Node.js, PM2"
fi
echo
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}" 