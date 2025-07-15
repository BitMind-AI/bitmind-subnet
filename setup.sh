#!/bin/bash
###########################################
# System Updates and Package Installation #
###########################################

# Update system
sudo apt update -y

# Install core dependencies
sudo apt install -y \
    python3-pip \
    nano \
    libgl1 \
    ffmpeg \
    unzip

"""
# Remove old nodejs and npm if present
#sudo apt-get remove --purge -y nodejs npm

# Install Node.js 20.x (LTS) from NodeSource for stability and universal standard
# NOTE: Update the version here when a new LTS is released
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs


# Install build dependencies
sudo apt install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev

# Install process manager (pm2) globally
sudo npm install -g pm2@latest
"""
############################
# Python Package Installation
############################

pip install --use-pep517 -e . -r requirements-git.txt
