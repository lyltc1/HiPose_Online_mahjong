ARG from=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
FROM ${from}

ENV DEBIAN_FRONTEND=noninteractive 
ENV DEBIAN_FRONTEND=teletype
ENV NVIDIA_VISIBLE_DEVICES=all 
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install normalSpeed
RUN apt-get update
RUN apt-get install -y build-essential unzip git cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
WORKDIR /home
COPY opencv-3.4.18.zip /home/opencv-3.4.18.zip
COPY opencv_contrib-3.4.18.zip /home/opencv_contrib-3.4.18.zip
RUN unzip opencv-3.4.18.zip
RUN unzip opencv_contrib-3.4.18.zip
WORKDIR /home/opencv-3.4.18/build
RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib-3.4.18/modules/ ..
RUN make -j7 && make install
COPY normalSpeed /home/normalSpeed
WORKDIR /home/normalSpeed/normalSpeed
RUN pip3 install "pybind11[global]"
RUN python3 setup.py install

# Install other dependency
RUN apt-get install -y libgl1
RUN pip install tqdm pandas scikit-learn open3d-cpu timm ultralytics

# Copy hipose and install RandLA
COPY hipose/models /home/hipose/models
WORKDIR /home/hipose/models/RandLA
RUN sh compile_op.sh

WORKDIR /home/HiPose_Online_mahjong/

# # Clean everything
# RUN apt-get clean
# RUN rm -rf /var/lib/apt/lists/*
# WORKDIR /home
# RUN rm opencv-3.4.18.zip
# RUN rm opencv_contrib-3.4.18.zip
# RUN rm -r opencv-3.4.18
# RUN rm -r opencv_contrib-3.4.18

