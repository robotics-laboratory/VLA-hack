#!/bin/bash
set -e

echo "=== LeRobot Workshop Setup ==="

ASSET_ARCHIVE="asset.zip"
DEFAULT_ASSET_GDRIVE_URL="https://drive.google.com/file/d/1YuvTRgxeehHA2cwGlMWSYxq2bzLMqNvL/view?usp=share_link"

# Install Python 3.12 and tkinter if needed
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing Python dependencies..."
    sudo apt update
    sudo apt install -y python3 python3-venv python3-tk unzip build-essential python3.12-dev
    sudo usermod -aG dialout $USER
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v python3.12 &> /dev/null; then
        echo "Installing Python 3.12..."
        brew install python@3.12
    fi
    brew install python-tk@3.12
    brew install unzip
fi

# Create virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
echo "Virtual environment created with $(python --version)"

# Install bootstrap tools first so assets can be fetched before requirements
pip install --upgrade pip
pip install gdown

# Download and unzip assets if not already present
if [ ! -d "asset" ]; then
    if [ -n "${ASSET_GDRIVE_URL:-}" ]; then
        echo "Downloading assets from Google Drive URL..."
        python -m gdown --fuzzy "${ASSET_GDRIVE_URL}" -O "${ASSET_ARCHIVE}"
    elif [ -n "${ASSET_GDRIVE_ID:-}" ]; then
        echo "Downloading assets from Google Drive file ID..."
        python -m gdown --id "${ASSET_GDRIVE_ID}" -O "${ASSET_ARCHIVE}"
    else
        echo "Downloading assets from default Google Drive URL..."
        python -m gdown --fuzzy "${DEFAULT_ASSET_GDRIVE_URL}" -O "${ASSET_ARCHIVE}"
    fi

    echo "Unzipping assets..."
    unzip -q -o "${ASSET_ARCHIVE}"
    echo "Assets extracted"
else
    echo "Assets already present, skipping download"
fi

# Install all dependencies (including lerobot)
pip install -r requirements.txt
echo "Dependencies installed"

echo ""
echo "=== Done! ==="
echo "To activate the environment: source .venv/bin/activate"
