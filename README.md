# Grounded SAM2 RealSense 3D Coordinate Demo for Windows

This repository is a Windows-adapted camera demo built on GroundingDINO and
SAM2. It provides live text-prompted detection, segmentation, continuous object
IDs, and per-object 3D coordinate estimation from an Intel RealSense depth
camera.

The main difference from a normal Grounded SAM2 camera demo is that this version
uses RealSense depth to estimate and display each tracked object's `(X, Y, Z)`
coordinate in meters.

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
