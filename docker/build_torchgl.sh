#!/bin/bash
set -euo pipefail

# Clone the NVIDIA OpenGL container images repository
if [ ! -d "./opengl" ]; then
    git clone https://gitlab.com/nvidia/container-images/opengl.git
    cd opengl
    git checkout ubuntu22.04
    cd ..
    cp opengl/NGC-DL-CONTAINER-LICENSE opengl/base/
fi

# Default parameters
BASE_IMAGE="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel"

# Check if the images already exist
if ! docker image inspect "lyltc1/pytorchgl/intermediate:base" > /dev/null 2>&1; then
    docker build -t "lyltc1/pytorchgl/intermediate:base" \
        --build-arg from=$BASE_IMAGE \
        "opengl/base"
else
    echo "Image lyltc1/pytorchgl/intermediate:base already exists. Skipping build."
fi

if ! docker image inspect "lyltc1/pytorchgl/intermediate:runtime" > /dev/null 2>&1; then
    docker build -t "lyltc1/pytorchgl/intermediate:runtime" \
        --build-arg from="lyltc1/pytorchgl/intermediate:base" \
        --build-arg LIBGLVND_VERSION="v1.2.0" \
        "opengl/glvnd/runtime"
else
    echo "Image lyltc1/pytorchgl/intermediate:runtime already exists. Skipping build."
fi

if ! docker image inspect "lyltc1/pytorchgl:2.1.2-cuda12.1-cudnn8-devel" > /dev/null 2>&1; then
    docker build -t "lyltc1/pytorchgl:2.1.2-cuda12.1-cudnn8-devel" \
        --build-arg from="lyltc1/pytorchgl/intermediate:runtime" \
        "opengl/glvnd/devel"
else
    echo "Image lyltc1/pytorchgl:2.1.2-cuda12.1-cudnn8-devel already exists. Skipping build."
fi
