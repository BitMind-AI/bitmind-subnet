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
    npm \
    ffmpeg \
    unzip

# Install build dependencies
sudo apt install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev

# Install process manager
sudo npm install -g pm2@latest

############################
# Python Package Installation
############################

pip install -e .
pip install -r requirements.txt
