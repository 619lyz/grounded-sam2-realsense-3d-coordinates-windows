# Grounded SAM2 RealSense 3D Coordinate Demo for Windows

This repository is a Windows-adapted camera demo built on GroundingDINO and
SAM2. It provides live text-prompted detection, segmentation, continuous object
IDs, and per-object 3D coordinate estimation from an Intel RealSense depth
camera.

The main difference from a normal Grounded SAM2 camera demo is that this version
uses RealSense depth to estimate and display each tracked object's `(X, Y, Z)`
coordinate in meters.

> **Compatibility note:** This project was originally developed on Linux. This
> repository contains a Windows adaptation, but Windows compatibility is not
> guaranteed to be perfect across all CUDA, Python, camera-driver, and hardware
> combinations.

## Upstream Projects and Credits

This repository is an engineering adaptation built from the following open
source projects:

- Grounded SAM 2 by IDEA-Research:
  <https://github.com/IDEA-Research/Grounded-SAM-2>
- SAM 2 by Meta:
  <https://github.com/facebookresearch/sam2>
- GroundingDINO by IDEA-Research:
  <https://github.com/IDEA-Research/GroundingDINO>

The original upstream Grounded SAM2 / SAM2 README content has been removed from
this fork so this README focuses only on this Windows RealSense 3D-coordinate
version. The original licenses are kept in this repository:

- `LICENSE`
- `LICENSE_sam2`
- `LICENSE_groundingdino`

## Important: This Repo Does Not Run Immediately After Clone

This repository does not include the SAM2 checkpoint file:

```text
checkpoints/sam2.1_hiera_large.pt
```

That file is about 900 MB, so it is intentionally ignored by Git. If you clone
this repository and run `main.py` without downloading the checkpoint first, the
program will fail when SAM2 tries to load the missing model file.

You must do all of the following before running:

1. Create the conda environment.
2. Install PyTorch and the Windows dependencies.
3. Download `sam2.1_hiera_large.pt`.
4. Place it under `checkpoints/`.
5. Connect a RealSense D455, or use OpenCV camera mode.

This is normal for model-based computer vision projects. The code is in Git; the
large model weight must be downloaded separately.

## What This Version Changes

- Renamed the main entry point to `main.py`.
- Adapted OpenCV camera opening for Windows by trying DirectShow and Media
  Foundation backends instead of the Linux-only V4L2 backend.
- Tested and adapted the RealSense path for Intel RealSense D455.
- Added RealSense color/depth alignment and depth filtering.
- Added per-object 3D coordinate display in meters.
- Added a Tkinter live preview window with prompt editing.
- Added VS Code launch configurations for OpenCV camera mode and RealSense
  mode.
- Added `requirements-windows.txt`.
- Added `.gitignore` rules for model checkpoints, generated outputs, and Python
  cache files.
- Guarded CUDA setup so importing the tracking module is less fragile on
  Windows.

## Ubuntu to Windows Porting Notes

The original demo worked more naturally on Ubuntu because the typical Linux
setup has a simpler CUDA/Python/camera stack for these research projects. Moving
the same code to Windows exposed several practical migration issues.

### 1. Python Environment Confusion

On Windows, VS Code and PowerShell can easily use different Python interpreters.
For example, VS Code may show the correct conda environment while the terminal
still runs a system Python such as:

```text
C:/Users/<user>/AppData/Local/Programs/Python/Python39/python.exe
```

This caused missing-module errors such as:

```text
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'hydra'
```

The fix is to run with the conda environment explicitly:

```powershell
E:\anaconda3\envs\dinosam2\python.exe main.py --source realsense
```

or ensure VS Code is using:

```text
E:\anaconda3\envs\dinosam2\python.exe
```

### 2. Linux V4L2 Camera Backend Does Not Work on Windows

The earlier camera opening code used:

```python
cv2.CAP_V4L2
```

That backend is for Linux. On Windows it can fail or behave inconsistently. This
version now tries Windows-friendly backends:

```python
cv2.CAP_DSHOW
cv2.CAP_MSMF
cv2.CAP_ANY
```

This matters mostly for the OpenCV fallback camera mode.

### 3. RealSense Requires the Windows SDK Path to Work

The project uses `pyrealsense2`. On Windows, the RealSense driver stack and
Python package must both work. This version was tested with:

```text
Intel RealSense D455
pyrealsense2
Windows
```

Before running the full model, it is useful to confirm that the camera is
visible:

```powershell
python -c "import pyrealsense2 as rs; print(len(rs.context().query_devices()))"
```

### 4. CUDA Import and Runtime Are More Fragile on Windows

The original tracking module configured CUDA autocast immediately at import
time. This is brittle on Windows, especially when debugging environment issues.
This version checks CUDA availability before enabling CUDA-specific setup.

The model still expects a CUDA GPU for practical speed. CPU mode is not the
target path for this demo.

### 5. Large Checkpoints Are Not Stored in Git

The local development folder contains:

```text
checkpoints/sam2.1_hiera_large.pt
```

but GitHub does not. This is intentional. The `.gitignore` file excludes:

```text
checkpoints/*.pt
checkpoints/*.pth
```

Without this rule, the repository would become very large and GitHub may reject
the file because normal GitHub repositories do not handle large binary model
weights well.

### 6. Performance Differs Greatly from a High-End Ubuntu GPU

The demo can run on an RTX 3060 Laptop GPU, but it is much slower than a high-end
Ubuntu workstation with an RTX 4090 or RTX 5090. SAM2 large plus GroundingDINO
is heavy, and RealSense depth processing adds extra overhead.

For Windows laptops, lower resolution and lower detection frequency are often
needed:

```powershell
python main.py --source realsense --width 424 --height 240 --fps 15 --detection-interval 60
```

## Supported Cameras

This version is intended for:

- Intel RealSense D455 in RealSense mode.
- Normal Windows USB cameras or laptop cameras in OpenCV mode.

RealSense mode uses:

- RGB color frames.
- Depth frames.
- RealSense camera intrinsics.
- Depth aligned to the color frame.

OpenCV mode only provides RGB images. It does not provide 3D coordinates because
there is no depth stream.

## Features

- Text-prompted object detection with GroundingDINO.
- SAM2 mask segmentation from detected boxes.
- SAM2 video tracking between detection frames.
- Continuous object IDs across frames.
- RealSense depth-based `(X, Y, Z)` coordinate estimation.
- Coordinate smoothing and jump limiting.
- Live overlay with mask, bounding box, object ID, class name, and 3D position.

Example overlay:

```text
1: person (0.12, -0.04, 1.36) m
2: cup    (0.31,  0.08, 0.74) m
```

The coordinates are in the RealSense camera coordinate system and are measured
in meters.

## Pipeline

```text
RealSense / OpenCV frame
        |
        v
GroundingDINO text-prompt detection
        |
        v
SAM2 image segmentation
        |
        v
SAM2 video propagation and tracking
        |
        v
mask + class name + tracking id
        |
        v
RealSense depth sampling
        |
        v
3D coordinate estimation and temporal smoothing
        |
        v
Tkinter live visualization
```

## Detection and Tracking

The detection model is loaded through HuggingFace:

```text
IDEA-Research/grounding-dino-tiny
```

The user prompt is normalized before use. For example:

```text
person.
hand.
cup.
```

The core tracking class is:

```python
IncrementalObjectTracker
```

It is implemented in:

```text
grounded_sam2_tracking_camera_with_continuous_id.py
```

The tracker runs GroundingDINO every `detection_interval` frames. Between those
frames, SAM2 video prediction propagates the existing masks. Continuous IDs are
maintained with:

- Class-name matching.
- Mask IoU matching.
- Bounding-box center distance as a fallback.
- Short-term missing-track tolerance before deleting an object.

## 3D Coordinate Strategy

3D coordinate estimation is implemented in `main.py`. The default mode is:

```text
pointcloud_median
```

This mode estimates the object coordinate from reliable depth points inside the
SAM2 mask.

### 1. Use Only Pixels Inside the Segmentation Mask

Depth points outside the SAM2 mask are ignored. This prevents obvious background
pixels from entering the coordinate estimate.

### 2. Prefer the Inner Mask Region

Object boundaries often contain mixed foreground and background depth. The code
uses a distance transform:

```python
cv2.distanceTransform(...)
```

It keeps the inner part of the mask when enough pixels are available. This makes
the coordinate estimate less sensitive to mask edges, tables, hands, or
background leakage.

### 3. Reject Invalid Depth

The default valid depth range is:

```python
MIN_VALID_DEPTH_M = 0.40
MAX_VALID_DEPTH_M = 4.0
```

Depth values outside this range are ignored. This avoids RealSense near-range
failure and far-range noise.

### 4. Select a Reliable Depth Cluster

A single mask can contain multiple depth layers. For example, the mask edge may
include the table behind an object. The function:

```python
select_depth_cluster(...)
```

groups depth values into bins and selects the most reliable foreground depth
cluster. If a previous valid 3D coordinate exists, the code prefers a cluster
near the previous depth. Otherwise, it prefers a strong near-depth cluster.

### 5. Select a Connected Depth Component

After depth clustering, the code performs connected-component analysis in image
space:

```python
select_connected_depth_component(...)
```

If the previous 3D point can be projected into the current frame, the component
containing that projected point is preferred. Otherwise, the largest connected
component is used.

### 6. Use Median Values for Robustness

The final depth value is the median of the selected depth values:

```text
Z = median(selected_depth_values)
```

The representative pixel `(u, v)` is also based on the median image position of
the selected points. Median statistics reduce the effect of outliers.

### 7. Deproject Pixel and Depth to 3D

The 2D pixel and depth are converted into a 3D point with RealSense intrinsics:

```text
X = (u - ppx) / fx * Z
Y = (v - ppy) / fy * Z
Z = depth
```

When `pyrealsense2` is available, the code uses:

```python
rs.rs2_deproject_pixel_to_point(...)
```

### 8. Smooth Coordinates Over Time

Each tracked object keeps a small history of recent 3D points. The estimator:

- Uses median history values.
- Applies exponential smoothing with `--xyz-alpha`.
- Limits sudden jumps with `--xyz-max-jump`.
- Can lock a relative point inside the object box to reduce drifting.

Useful parameters:

```text
--xyz-alpha
--xyz-max-jump
--xyz-history-size
--xyz-min-history
--no-lock-center
```

## Installation on Windows

Create a conda environment:

```powershell
conda create -n dinosam2 python=3.10 pip -y
conda activate dinosam2
```

Install PyTorch with CUDA 12.1 wheels:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install this project and its Windows runtime dependencies:

```powershell
python -m pip install --no-build-isolation -e . -r requirements-windows.txt
```

## Model Checkpoint

The SAM2 `.pt` checkpoint is not committed to this repository because it is
large. After cloning this repository, the checkpoint directory only contains the
download script. The default code expects this exact file path:

```text
checkpoints/sam2.1_hiera_large.pt
```

If this file is missing, the repository is installed correctly but the demo is
not ready to run yet.

Download it from the official Meta public file server:

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Place it here:

```text
checkpoints/sam2.1_hiera_large.pt
```

You can also use the included script:

```bash
cd checkpoints
bash download_ckpts.sh
```

The script downloads several SAM2.1 checkpoints. This demo uses the large
checkpoint by default.

Do not rename the file unless you also update the `sam2_ckpt_path` argument in
the code.

## Run

RealSense mode:

```powershell
python main.py --source realsense
```

OpenCV camera mode:

```powershell
python main.py --source opencv --camera 0
```

If the wrong camera opens, try another index:

```powershell
python main.py --source opencv --camera 1
```

For slower GPUs, reduce camera load and run full detection less often:

```powershell
python main.py --source realsense --width 424 --height 240 --fps 15 --detection-interval 60
```

## Common Parameters

```text
--source realsense|opencv
--camera 0
--prompt person.
--detection-interval 20
--box-threshold 0.35
--text-threshold 0.30
--width 640
--height 480
--fps 30
--depth-mode pointcloud_median|mask_mean|locked_point
```

## File Overview

```text
main.py
```

Main application. It handles the GUI, camera input, RealSense depth frames, 3D
coordinate estimation, smoothing, and drawing.

```text
grounded_sam2_tracking_camera_with_continuous_id.py
```

GroundingDINO + SAM2 detection, segmentation, tracking, and continuous-ID logic.

```text
sam2/
```

SAM2 model code and configuration files.

```text
utils/
```

Mask dictionaries, tracking helpers, and visualization utilities.

```text
checkpoints/
```

Model checkpoint directory. `.pt` files are ignored by Git and must be downloaded
separately.

## VS Code

The repository includes `.vscode/launch.json` with:

```text
Run main demo (OpenCV camera)
Run main demo (RealSense)
```

Select the `dinosam2` interpreter in VS Code, or point VS Code directly to:

```text
E:\anaconda3\envs\dinosam2\python.exe
```

## Performance Notes

SAM2 large plus GroundingDINO is heavy. It can run on an RTX 3060 Laptop, but it
will be much slower than on a high-end GPU such as an RTX 4090 or RTX 5090.

If the demo is slow:

- Lower the RealSense resolution.
- Lower the FPS.
- Increase `--detection-interval`.
- Use a smaller SAM2 checkpoint such as tiny or small.

## Credits

This is a Windows adaptation and engineering cleanup around Grounded SAM2 and
SAM2. The original models and upstream code belong to their respective authors.

---

# Grounded SAM2 RealSense 3D Coordinate Demo for Windows 中文版

本仓库是一个适配 Windows 的相机演示项目，基于 GroundingDINO 和 SAM2，实现实时文本提示目标检测、目标分割、连续目标 ID，以及基于 Intel RealSense 深度相机的每个目标 3D 坐标估计。

普通 Grounded SAM2 相机 demo 主要做检测、分割和跟踪；本项目额外使用 RealSense 深度信息，估计并显示每个被跟踪目标的 `(X, Y, Z)` 坐标，单位为米。

> **兼容性说明：** 本项目最初是在 Linux 环境下开发的。本仓库虽然对 Windows 做了适配，但由于 CUDA、Python、相机驱动和硬件组合差异较大，不保证在所有 Windows 环境中都能完美适配。

## 上游项目与致谢

本仓库是在以下开源项目基础上做的工程适配：

- IDEA-Research 的 Grounded SAM 2:
  <https://github.com/IDEA-Research/Grounded-SAM-2>
- Meta 的 SAM 2:
  <https://github.com/facebookresearch/sam2>
- IDEA-Research 的 GroundingDINO:
  <https://github.com/IDEA-Research/GroundingDINO>

本 fork 移除了上游 Grounded SAM2 / SAM2 的原始长 README 内容，使 README 只聚焦于这个 Windows RealSense 3D 坐标版本。原始许可证仍保留在仓库中：

- `LICENSE`
- `LICENSE_sam2`
- `LICENSE_groundingdino`

## 重要说明：Clone 后不能直接运行

本仓库不包含 SAM2 权重文件：

```text
checkpoints/sam2.1_hiera_large.pt
```

该文件约 900 MB，因此被有意排除在 Git 之外。如果 clone 仓库后没有先下载 checkpoint 就运行 `main.py`，程序会在 SAM2 加载模型时因为缺少权重文件而失败。

运行前需要完成以下步骤：

1. 创建 conda 环境。
2. 安装 PyTorch 和 Windows 依赖。
3. 下载 `sam2.1_hiera_large.pt`。
4. 将其放到 `checkpoints/` 目录下。
5. 连接 RealSense D455，或使用 OpenCV 摄像头模式。

这是模型类计算机视觉项目中的正常情况：代码保存在 Git 中，大模型权重需要单独下载。

## 这个版本做了哪些改动

- 将主入口重命名为 `main.py`。
- 针对 Windows 调整 OpenCV 摄像头打开方式，尝试 DirectShow 和 Media Foundation 后端，而不是 Linux 专用的 V4L2。
- 针对 Intel RealSense D455 测试并适配 RealSense 路径。
- 增加 RealSense 彩色图和深度图对齐，以及深度滤波。
- 增加每个目标的 3D 坐标显示，单位为米。
- 增加带 prompt 编辑功能的 Tkinter 实时预览窗口。
- 增加 VS Code 启动配置，支持 OpenCV 摄像头模式和 RealSense 模式。
- 增加 `requirements-windows.txt`。
- 增加 `.gitignore`，忽略模型权重、生成输出和 Python 缓存文件。
- 对 CUDA 初始化做保护，降低 Windows 上 import tracking 模块时的脆弱性。

## Ubuntu 到 Windows 的移植说明

原始 demo 在 Ubuntu 上更自然，因为 Linux 通常更适合这类研究项目中的 CUDA、Python 和相机栈。迁移到 Windows 后，会暴露一些实际工程问题。

### 1. Python 环境容易混乱

在 Windows 上，VS Code 和 PowerShell 可能使用不同的 Python 解释器。例如 VS Code 显示的是正确 conda 环境，但终端实际运行的是系统 Python：

```text
C:/Users/<user>/AppData/Local/Programs/Python/Python39/python.exe
```

这会导致缺模块错误，例如：

```text
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'hydra'
```

解决方法是显式使用 conda 环境运行：

```powershell
E:\anaconda3\envs\dinosam2\python.exe main.py --source realsense
```

或者确认 VS Code 使用的是：

```text
E:\anaconda3\envs\dinosam2\python.exe
```

### 2. Linux V4L2 摄像头后端不能在 Windows 上使用

早期代码使用：

```python
cv2.CAP_V4L2
```

这是 Linux 摄像头后端，在 Windows 上可能失败或表现不稳定。本版本改为尝试 Windows 友好的后端：

```python
cv2.CAP_DSHOW
cv2.CAP_MSMF
cv2.CAP_ANY
```

这主要影响 OpenCV fallback 摄像头模式。

### 3. RealSense 在 Windows 上需要驱动和 SDK 路径正常

项目使用 `pyrealsense2`。在 Windows 上，RealSense 驱动栈和 Python 包都需要正常工作。本版本测试环境包括：

```text
Intel RealSense D455
pyrealsense2
Windows
```

运行完整模型前，可以先确认相机是否可见：

```powershell
python -c "import pyrealsense2 as rs; print(len(rs.context().query_devices()))"
```

### 4. Windows 上 CUDA import 和运行更脆弱

原 tracking 模块会在 import 时立即配置 CUDA autocast，这在 Windows 调试环境问题时比较脆弱。本版本会先检查 CUDA 是否可用，再启用 CUDA 相关设置。

模型实际运行仍然建议使用 CUDA GPU。CPU 不是本 demo 的目标运行路径。

### 5. 大模型权重不存入 Git

本地开发目录中有：

```text
checkpoints/sam2.1_hiera_large.pt
```

但 GitHub 仓库中没有，这是有意设计。`.gitignore` 会排除：

```text
checkpoints/*.pt
checkpoints/*.pth
```

否则仓库会非常大，并且 GitHub 普通仓库可能会拒绝这类大二进制模型权重。

### 6. 性能和高端 Ubuntu GPU 差异很大

这个 demo 可以在 RTX 3060 Laptop GPU 上运行，但会明显慢于配备 RTX 4090 或 RTX 5090 的高端 Ubuntu 工作站。SAM2 large + GroundingDINO 本身较重，RealSense 深度处理也会增加开销。

Windows 笔记本上通常需要降低分辨率并减少 DINO 全量检测频率：

```powershell
python main.py --source realsense --width 424 --height 240 --fps 15 --detection-interval 60
```

## 支持的相机

本版本主要面向：

- RealSense 模式下的 Intel RealSense D455。
- OpenCV 模式下的普通 Windows USB 摄像头或笔记本摄像头。

RealSense 模式使用：

- RGB 彩色帧。
- 深度帧。
- RealSense 相机内参。
- 对齐到彩色图的深度图。

OpenCV 模式只提供 RGB 图像，不提供 3D 坐标，因为没有深度流。

## 功能

- 使用 GroundingDINO 做文本提示目标检测。
- 使用 SAM2 根据检测框生成 mask。
- 使用 SAM2 在检测帧之间做视频跟踪。
- 跨帧保持连续目标 ID。
- 基于 RealSense 深度估计 `(X, Y, Z)` 坐标。
- 坐标平滑和跳变限制。
- 实时叠加 mask、检测框、目标 ID、类别名和 3D 坐标。

示例显示：

```text
1: person (0.12, -0.04, 1.36) m
2: cup    (0.31,  0.08, 0.74) m
```

坐标位于 RealSense 相机坐标系中，单位为米。

## 技术流程

```text
RealSense / OpenCV frame
        |
        v
GroundingDINO text-prompt detection
        |
        v
SAM2 image segmentation
        |
        v
SAM2 video propagation and tracking
        |
        v
mask + class name + tracking id
        |
        v
RealSense depth sampling
        |
        v
3D coordinate estimation and temporal smoothing
        |
        v
Tkinter live visualization
```

## 检测和跟踪

检测模型通过 HuggingFace 加载：

```text
IDEA-Research/grounding-dino-tiny
```

用户 prompt 会在使用前进行规范化。例如：

```text
person.
hand.
cup.
```

核心 tracking 类是：

```python
IncrementalObjectTracker
```

实现文件是：

```text
grounded_sam2_tracking_camera_with_continuous_id.py
```

tracker 每隔 `detection_interval` 帧运行一次 GroundingDINO。在两次检测之间，SAM2 video prediction 会传播已有 mask。连续 ID 通过以下方式维护：

- 类别名匹配。
- mask IoU 匹配。
- 检测框中心距离作为 fallback。
- 短期丢失容忍，超过后删除目标。

## 3D 坐标策略

3D 坐标估计在 `main.py` 中实现。默认模式是：

```text
pointcloud_median
```

该模式从 SAM2 mask 内可靠的深度点估计目标坐标。

### 1. 只使用分割 mask 内的像素

SAM2 mask 外的深度点会被忽略，避免明显背景点进入坐标估计。

### 2. 优先使用 mask 内部区域

目标边界经常混入前景和背景深度。代码使用距离变换：

```python
cv2.distanceTransform(...)
```

当有足够像素时，会保留 mask 内部区域，使坐标估计对边缘、桌面、手部或背景泄漏更不敏感。

### 3. 过滤无效深度

默认有效深度范围：

```python
MIN_VALID_DEPTH_M = 0.40
MAX_VALID_DEPTH_M = 4.0
```

超出该范围的深度值会被忽略，用来避免 RealSense 近距离失效和远距离噪声。

### 4. 选择可靠深度簇

同一个 mask 内可能包含多个深度层。例如 mask 边缘可能包含物体后方的桌面。函数：

```python
select_depth_cluster(...)
```

会把深度值分桶，并选择最可靠的前景深度簇。如果已有上一次有效 3D 坐标，代码会优先选择接近上一帧深度的簇；否则优先选择强的近深度簇。

### 5. 选择连通深度区域

深度聚类后，代码在图像空间做连通区域分析：

```python
select_connected_depth_component(...)
```

如果上一帧 3D 点能投影到当前图像中，则优先使用包含该投影点的连通区域；否则使用最大的连通区域。

### 6. 使用中位数提高鲁棒性

最终深度值使用选中深度点的中位数：

```text
Z = median(selected_depth_values)
```

代表像素 `(u, v)` 也基于选中点的图像位置中位数。中位数统计可以减少离群点影响。

### 7. 将像素和深度反投影到 3D

使用 RealSense 内参将 2D 像素和深度转换为 3D 点：

```text
X = (u - ppx) / fx * Z
Y = (v - ppy) / fy * Z
Z = depth
```

当 `pyrealsense2` 可用时，代码使用：

```python
rs.rs2_deproject_pixel_to_point(...)
```

### 8. 对坐标进行时间平滑

每个跟踪目标都会保留一小段最近 3D 点历史。估计器会：

- 使用历史中位数。
- 通过 `--xyz-alpha` 做指数平滑。
- 通过 `--xyz-max-jump` 限制突然跳变。
- 可以锁定目标框内的相对点，减少漂移。

常用参数：

```text
--xyz-alpha
--xyz-max-jump
--xyz-history-size
--xyz-min-history
--no-lock-center
```

## Windows 安装

创建 conda 环境：

```powershell
conda create -n dinosam2 python=3.10 pip -y
conda activate dinosam2
```

安装 CUDA 12.1 版本 PyTorch：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装本项目和 Windows 运行依赖：

```powershell
python -m pip install --no-build-isolation -e . -r requirements-windows.txt
```

## 模型权重

SAM2 `.pt` 权重文件没有提交到本仓库，因为文件很大。clone 仓库后，checkpoint 目录只包含下载脚本。默认代码需要这个精确路径：

```text
checkpoints/sam2.1_hiera_large.pt
```

如果该文件缺失，说明仓库安装可能没问题，但 demo 还没有准备好运行。

从 Meta 官方公开文件服务器下载：

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

放置到：

```text
checkpoints/sam2.1_hiera_large.pt
```

也可以使用仓库脚本：

```bash
cd checkpoints
bash download_ckpts.sh
```

该脚本会下载多个 SAM2.1 checkpoint。本 demo 默认使用 large checkpoint。

除非同时修改代码中的 `sam2_ckpt_path` 参数，否则不要重命名该文件。

## 运行

RealSense 模式：

```powershell
python main.py --source realsense
```

OpenCV 摄像头模式：

```powershell
python main.py --source opencv --camera 0
```

如果打开了错误摄像头，可以尝试其他 index：

```powershell
python main.py --source opencv --camera 1
```

对于较慢 GPU，可以降低相机负载并减少全量检测频率：

```powershell
python main.py --source realsense --width 424 --height 240 --fps 15 --detection-interval 60
```

## 常用参数

```text
--source realsense|opencv
--camera 0
--prompt person.
--detection-interval 20
--box-threshold 0.35
--text-threshold 0.30
--width 640
--height 480
--fps 30
--depth-mode pointcloud_median|mask_mean|locked_point
```

## 文件说明

```text
main.py
```

主程序，负责 GUI、相机输入、RealSense 深度帧、3D 坐标估计、平滑和绘制。

```text
grounded_sam2_tracking_camera_with_continuous_id.py
```

GroundingDINO + SAM2 检测、分割、跟踪和连续 ID 逻辑。

```text
sam2/
```

SAM2 模型代码和配置文件。

```text
utils/
```

mask 字典、跟踪辅助函数和可视化工具。

```text
checkpoints/
```

模型权重目录。`.pt` 文件被 Git 忽略，需要单独下载。

## VS Code

仓库包含 `.vscode/launch.json`，其中有：

```text
Run main demo (OpenCV camera)
Run main demo (RealSense)
```

请在 VS Code 中选择 `dinosam2` 解释器，或直接指向：

```text
E:\anaconda3\envs\dinosam2\python.exe
```

## 性能说明

SAM2 large + GroundingDINO 计算量较大。它可以在 RTX 3060 Laptop 上运行，但会比 RTX 4090 或 RTX 5090 等高端 GPU 慢很多。

如果 demo 较慢：

- 降低 RealSense 分辨率。
- 降低 FPS。
- 增大 `--detection-interval`。
- 使用更小的 SAM2 checkpoint，例如 tiny 或 small。

## 致谢

这是围绕 Grounded SAM2 和 SAM2 做的 Windows 适配与工程整理版本。原始模型和上游代码归各自作者所有。
