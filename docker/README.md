# Run Code in Docker

## Download the image
We provide an image that can be downloaded using the command below.
```
docker pull lyltc1/hipose_online:latest
```
## Build images
Options: you can build the image by yourself.

```bash
cd HiPose/docker
bash build_docker.sh
```

## Usage
Pay attention to the dataset and output volume.
```
HiPose_Online_dir=~/git/HiPose_Online_mahjong

docker run -it \
--privileged \
--gpus all \
--shm-size 12G \
--runtime=nvidia \
--env=DISPLAY=$DISPLAY \
--env="XAUTHORITY=${XAUTH}" \
--env="QT_X11_NO_MITSHM=1" \
--env="LIBGL_ALWAYS_INDIRECT=" \
--env="LIBGL_ALWAYS_SOFTWARE=1" \
--rm \
-v /etc/group:/etc/group:ro \
-v /etc/passwd:/etc/passwd:ro \
-v ${HiPose_Online_dir}:/home/HiPose_Online_mahjong \
--net host \
lyltc1/hipose_online:mahjong-1.0
```

