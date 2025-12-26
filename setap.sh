#!/bin/bash
set -e

mkdir -p workspace/imageClassifier
cd workspace/imageClassifier

apt-get update -y
apt-get install -y python3-venv wget curl

# Create venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch CUDA
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# Install Python deps
pip install \
  transformers==4.35.2 \
  accelerate==0.24.1 \
  flask \
  pillow \
  requests[socks]

# Download files
wget -O imageSplit.py https://raw.githubusercontent.com/bertiesknapp15-lab/as/refs/heads/main/imageSplit.py
wget -O run.py https://raw.githubusercontent.com/bertiesknapp15-lab/as/refs/heads/main/run.py

# Run app
nohup venv/bin/python run.py

