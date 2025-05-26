#!/bin/bash
set -euo pipefail

# Navigate to the directory containing the build_torchgl.sh script and execute it
if [ -f "./build_torchgl.sh" ]; then
    chmod +x ./build_torchgl.sh
    bash ./build_torchgl.sh
else
    echo "Error: build_torchgl.sh not found in the directory."
    exit 1
fi

# manually download opencv = 3.4.18 from https://github.com/opencv/opencv/archive/refs/tags/3.4.18.zip
if [ ! -f "./opencv-3.4.18.zip" ]; then
    wget https://github.com/opencv/opencv/archive/refs/tags/3.4.18.zip -O opencv-3.4.18.zip
fi  
# manually download opencv_contrib = 3.4.18 from https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.18.zip
if [ ! -f "./opencv_contrib-3.4.18.zip" ]; then
    wget https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.18.zip -O opencv_contrib-3.4.18.zip
fi

# Check if the normalSpeed repository exists, otherwise clone it
if [ ! -d "./normalSpeed" ]; then
    git clone https://github.com/hfutcgncas/normalSpeed.git
fi

docker build -t lyltc1/hipose_online:mahjong-1.0 --build-arg from="lyltc1/pytorchgl:2.1.2-cuda12.1-cudnn8-devel" .
