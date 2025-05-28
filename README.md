# HiPose_Online_mahjong

[中文说明 (Chinese Version)](README_CN.md)

This repository is for object pose estimation demonstration, introducing a workflow for data generation, training, and deployment of object pose estimation. It is based on [ultralitc](https://github.com/uzh-rpg/ultralitc) and [HiPose](https://github.com/lyltc1/HiPose) projects, and is used for pose estimation of Sichuan Mahjong tiles.

> **All related code, models, scripts, and data can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/11zwgrygs7210UBUZWFpXxg?pwd=ai5t). The term [Cloud Address] below refers to this link.**

## Table of Contents

1. [Prepare Object Model Files](#prepare-object-model-files)
2. [Generate BOP Format Dataset](#generate-bop-format-dataset)
3. [Train Detection Model](#train-detection-model)
4. [Train Pose Estimation Model](#train-pose-estimation-model)
5. [Deployment](#deployment)

## Prepare Object Model Files

We provide the following model files:

- `[Cloud Address]/bop/wan7.zip`: Contains only the "7 Wan" Mahjong tile model
- `[Cloud Address]/bop/sichuan.zip`: Contains models for all Sichuan Mahjong tiles

The detection model will detect all Sichuan Mahjong tiles, while pose estimation is only for the "7 Wan" tile.

**Download steps:**

1. Download the `[Cloud Address]/bop` folder  
2. Unzip the files:
    ```bash
    unzip wan7.zip -d wan7
    unzip sichuan.zip -d sichuan
    ```

**After extraction, the directory structure is as follows:**
```
bop/
├── sichuan/
│   └── models/
├── wan7/
    └── models/
```

## Generate BOP Format Dataset

1. Install the BlenderProc rendering library. See the [BlenderProc](https://github.com/DLR-RM/BlenderProc) official documentation for installation instructions.
2. Download the `[Cloud Address]/blenderproc_script` script and copy it to the `blenderproc` folder.
3. In the `blenderproc` folder, run `blenderproc download cc_textures` to download texture resources.

**Example commands to generate datasets:**
```bash
# Generate 7 Wan dataset
blenderproc debug examples/datasets/bop_challenge/main_7wan_upright.py /path/to/bop/ resources/cctextures examples/datasets/bop_challenge/output --num_scenes=2000

# Generate Sichuan Mahjong (near distance) dataset
blenderproc debug examples/datasets/bop_challenge/main_sichuan_near.py /path/to/bop/ resources/cctextures examples/datasets/bop_challenge/output --num_scenes=2000

# Generate Sichuan Mahjong (far distance) dataset
blenderproc debug examples/datasets/bop_challenge/main_sichuan_far.py path/to/bop/ resources/cctextures examples/datasets/bop_challenge/output --num_scenes=2000
```

## Train Detection Model

Convert the BOP dataset to the format required by ultralitic. The conversion script is at `[Cloud Address]/ultralitic_script/data/detection_bop_converter.py`.

We also provide converted simulation and real images and labels at `[Cloud Address]/data/*train_dataset.zip`.

Refer to `[Cloud Address]/ultralitic_script/train_det.py` and `[Cloud Address]/ultralitic_script/test_det.py` for training and testing scripts.

The trained model file can be found at `[Cloud Address]/ckpt/yolo11_wan7.pt`.

## Train Pose Estimation Model

Refer to the [HiPose project](https://github.com/lyltc1/HiPose). It is recommended to use a Docker environment to avoid dependency issues. When training the wan7 dataset, replace files in the `HiPose` folder with those in `[Cloud Address]/HiPose_script` (mainly to add configuration files, etc.).
The `bop_toolkit` is installed; please replace files in the `HiPose` folder with those in `[Cloud Address]/bop_toolkit_script` as needed.

For detailed training and validation procedures, see the [HiPose project documentation](https://github.com/lyltc1/HiPose).

The trained model file can be found at `[Cloud Address]/ckpt/HiPose_wan7_0_9051step164000`.

## Deployment

This repository provides a Docker image for deployment. Camera parameters need to be configured in the container. The deployment code for ultralitic and HiPose is integrated, with training-related content removed. See `[Cloud Address]/HiPose_Online_mahjong/test_yolo.py` and `[Cloud Address]/HiPose_Online_mahjong/test_hipose_real_3D_video.py` for details.

Download test data (about 9.3G) from the `[Cloud Address]/data` folder. First, download the camera parameter files: `[Cloud Address]/data/camera.json` is for real data, and `[Cloud Address]/data/camera_pbr.json` is for simulated data. Then download the test videos: `[Cloud Address]/data/sequence_*.zip` are different sequences containing various test cases.

The following test code can read saved video sequences and output visualization results:
```bash
python test_yolo.py
python test_hipose_real_3D_video.py
```

The saved video results can be found in the `[Cloud Address]/output` folder.
