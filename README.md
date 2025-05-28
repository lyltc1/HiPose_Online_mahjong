# HiPose_Online_mahjong

本仓库用于物体位姿估计演示，介绍了一个实现物体位姿估计的数据生成/训练/部署的流程。基于 [ultralitc](https://github.com/uzh-rpg/ultralitc)以及[HiPose](https://github.com/lyltc1/HiPose) 项目，用于四川麻将牌位姿估计。

> **所有相关代码、模型、脚本和数据可在公司内网 NAS `/Home/prj-mt/林永良数据/` 下载。以下用【云盘地址】指代该地址**

## 目录

1. [准备物体模型文件](#准备物体模型文件)
2. [生成BOP格式数据集](#生成bop格式数据集)
3. [训练检测模型](#训练检测模型)
4. [训练位姿估计模型](#训练位姿估计模型)
5. [部署](#部署)

## 准备物体模型文件

我们提供以下模型文件：

- `云盘地址/bop/wan7.zip`：仅包含麻将牌“7万”的模型
- `云盘地址/bop/sichuan.zip`：包含四川麻将所有牌的模型

后续检测模型将对所有四川麻将牌进行检测，位姿估计仅针对“7万”牌。

**下载步骤：**

1. 下载 `云盘地址/bop` 文件夹  
3. 解压文件：
    ```bash
    unzip wan7.zip -d wan7
    unzip sichuan.zip -d sichuan
    ```

**解压后目录结构如下：**
```
bop/
├── sichuan/
│   └── models/
├── wan7/
    └── models/
```

## 生成BOP格式数据集

1. 安装 BlenderProc 渲染库，安装方法详见 [BlenderProc](https://github.com/DLR-RM/BlenderProc) 官方文档。
2. 下载 `云盘地址/blenderproc_script` 脚本并复制到`blenderproc`文件夹中。
3. 在`blenderproc`文件夹中运行 `blenderproc download cc_textures` 下载纹理资源。

**生成数据集命令示例：**
```bash
# 生成 7万 数据集
blenderproc debug examples/datasets/bop_challenge/main_7wan_upright.py /path/to/bop/ resources/cctextures examples/datasets/bop_challenge/output --num_scenes=2000

# 生成四川麻将（近距离）数据集
blenderproc debug examples/datasets/bop_challenge/main_sichuan_near.py /path/to/bop/ resources/cctextures examples/datasets/bop_challenge/output --num_scenes=2000

# 生成四川麻将（远距离）数据集
blenderproc debug examples/datasets/bop_challenge/main_sichuan_far.py path/to/bop/ resources/cctextures examples/datasets/bop_challenge/output --num_scenes=2000
```

## 训练检测模型

将 BOP 数据集转换为 ultralitic 所需格式，转换脚本见 `云盘地址/ultralitic_script/data/detection_bop_converter.py`。

训练和测试脚本参考 `云盘地址/ultralitic_script/train_det.py` 和 `云盘地址/ultralitic_script/test_det.py`。

训练完成的模型文件见 `云盘地址/ckpt/yolo11_wan7.pt`。

## 训练位姿估计模型

参考 [HiPose 项目](https://github.com/lyltc1/HiPose)，建议使用 Docker 环境以避免依赖问题。训练 wan7 数据集时，请将 `云盘地址/HiPose_script` 文件夹下的同名文件替换 `HiPose` 文件夹，主要是增加一些配置文件等。

训练及验证流程详见 [HiPose 项目文档](https://github.com/lyltc1/HiPose)。

训练完成的模型文件见 `云盘地址/ckpt/HiPose_wan7_0_9051step164000`。

## 部署

本仓库提供了部署用 Docker 镜像，需在容器中配置相机参数。已集成 ultralitic 和 HiPose 的部署代码，移除了训练相关内容。具体可参考 `云盘地址/HiPose_Online_mahjong/test_yolo.py` 和 `云盘地址/HiPose_Online_mahjong/test_hipose_real_3D_video.py`。

下载测试数据（约9.3G），见`云盘地址/data`文件夹。首先下载相机参数文件，`云盘地址/data/camera.json`是真实数据的相机内参，`云盘地址/data/camera_pbr.json`是仿真数据的相机内参。然后下载测试视频，`云盘地址/data/sequence_*.zip `是不同的序列，包含不同测试用例。


以下测试代码可读取保存的视频序列并输出可视化结果：
```bash
python test_yolo.py
python test_hipose_real_3D_video.py
```

保存的视频结果见 `云盘地址/output` 文件夹。