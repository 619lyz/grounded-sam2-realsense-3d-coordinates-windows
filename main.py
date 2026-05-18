import argparse
import queue
import platform
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


MIN_VALID_DEPTH_M = 0.40
MAX_VALID_DEPTH_M = 4.0


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.strip().lower()
    if not prompt:
        return "person."
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def open_camera(preferred_index: int):
    backends = [cv2.CAP_ANY]
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif platform.system() == "Linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(preferred_index, backend)
        if cap.isOpened():
            return cap, preferred_index
        cap.release()

    for index in range(8):
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap, index
            cap.release()

    return None, None


class RealSenseColorDepthCamera:
    def __init__(self, width: int, height: int, fps: int):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed.")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.intrinsics = None
        self.depth_scale = 0.001
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

    def start(self):
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return False, None, None

        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filling_filter.process(depth_frame)

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return True, color_rgb, depth_m

    def stop(self):
        self.pipeline.stop()


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def mask_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def box_center(box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)


def pixel_to_box_relative(pixel, box):
    if pixel is None or box is None:
        return None
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    u, v = pixel
    return ((u - x1) / width, (v - y1) / height)


def box_relative_to_pixel(relative, box):
    if relative is None or box is None:
        return None
    x1, y1, x2, y2 = box
    rel_x, rel_y = relative
    u = int(round(x1 + rel_x * max(x2 - x1, 1)))
    v = int(round(y1 + rel_y * max(y2 - y1, 1)))
    return u, v


def nearest_mask_pixel(mask: np.ndarray, target_pixel):
    target_u, target_v = target_pixel
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    distances = (xs - target_u) ** 2 + (ys - target_v) ** 2
    nearest = int(np.argmin(distances))
    return int(xs[nearest]), int(ys[nearest])


def deproject_pixel(intrinsics, pixel, depth_m):
    u, v = pixel
    if rs is not None:
        return np.asarray(
            rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth_m)),
            dtype=np.float64,
        )

    x = (float(u) - intrinsics.ppx) / intrinsics.fx * float(depth_m)
    y = (float(v) - intrinsics.ppy) / intrinsics.fy * float(depth_m)
    return np.array([x, y, float(depth_m)], dtype=np.float64)


def project_point(intrinsics, xyz, width, height):
    x, y, z = [float(value) for value in xyz]
    if z <= 0:
        return None
    u = int(round(intrinsics.fx * x / z + intrinsics.ppx))
    v = int(round(intrinsics.fy * y / z + intrinsics.ppy))
    return int(np.clip(u, 0, width - 1)), int(np.clip(v, 0, height - 1))


def local_depth_xyz(mask: np.ndarray, depth_m: np.ndarray, intrinsics, pixel, ksize=21):
    if depth_m is None or intrinsics is None or pixel is None:
        return None

    height, width = depth_m.shape[:2]
    mask = mask[:height, :width].astype(bool)
    u, v = pixel
    if not (0 <= u < width and 0 <= v < height):
        return None
    if not mask[v, u]:
        nearest = nearest_mask_pixel(mask, (u, v))
        if nearest is None:
            return None
        u, v = nearest

    half = ksize // 2
    samples = []
    for vv in range(max(0, v - half), min(height, v + half + 1)):
        for uu in range(max(0, u - half), min(width, u + half + 1)):
            if mask[vv, uu]:
                z = float(depth_m[vv, uu])
                if np.isfinite(z) and MIN_VALID_DEPTH_M < z < MAX_VALID_DEPTH_M:
                    samples.append(z)

    if len(samples) < 4:
        return None

    depths = np.array(samples, dtype=np.float64)
    median_depth = np.median(depths)
    mad = np.median(np.abs(depths - median_depth))
    mad = max(float(mad), 1e-6)
    threshold = max(0.02, 3.5 * 1.4826 * mad)
    kept = depths[np.abs(depths - median_depth) <= threshold]
    if kept.size >= 4:
        depths = kept

    z = float(np.median(depths))
    return (u, v), deproject_pixel(intrinsics, (u, v), z)


def mask_average_depth_xyz(mask: np.ndarray, depth_m: np.ndarray, intrinsics):
    if depth_m is None or intrinsics is None:
        return None

    height, width = depth_m.shape[:2]
    mask = mask[:height, :width].astype(bool)
    valid = mask & np.isfinite(depth_m) & (depth_m > MIN_VALID_DEPTH_M) & (depth_m < MAX_VALID_DEPTH_M)
    if valid.sum() < 8:
        return None

    ys, xs = np.where(valid)
    depths = depth_m[ys, xs].astype(np.float64)

    median_depth = np.median(depths)
    mad = np.median(np.abs(depths - median_depth))
    mad = max(float(mad), 1e-6)
    threshold = max(0.02, 3.5 * 1.4826 * mad)
    keep = np.abs(depths - median_depth) <= threshold
    if keep.sum() >= 8:
        xs = xs[keep]
        ys = ys[keep]
        depths = depths[keep]

    z = float(np.mean(depths))
    u = int(np.round(np.median(xs)))
    v = int(np.round(np.median(ys)))
    if not mask[v, u]:
        nearest = nearest_mask_pixel(mask, (u, v))
        if nearest is None:
            return None
        u, v = nearest

    return (u, v), deproject_pixel(intrinsics, (u, v), z)


def mask_median_raw_depth(mask: np.ndarray, depth_m: np.ndarray):
    if depth_m is None:
        return None

    height, width = depth_m.shape[:2]
    mask = mask[:height, :width].astype(np.uint8)
    if mask.sum() < 8:
        return None

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if distance.max() > 0:
        inner = (distance >= max(1.0, float(distance.max()) * 0.18)).astype(np.uint8)
        if inner.sum() >= 8:
            mask = inner

    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < MAX_VALID_DEPTH_M)
    if valid.sum() < 8:
        return None

    return float(np.median(depth_m[valid]))


def select_depth_cluster(xs, ys, z_values, prior_z=None, bin_size=0.025, prior_window=0.08):
    if z_values.size == 0:
        return xs, ys, z_values

    if prior_z is not None:
        keep = np.abs(z_values - prior_z) <= prior_window
        if keep.sum() >= 8:
            return xs[keep], ys[keep], z_values[keep]

    z_min = float(z_values.min())
    z_max = float(z_values.max())
    if z_max - z_min < bin_size:
        return xs, ys, z_values

    bins = np.arange(z_min, z_max + bin_size, bin_size)
    hist, edges = np.histogram(z_values, bins=bins)
    if hist.size == 0:
        return xs, ys, z_values

    max_count = hist.max()
    candidate_bins = np.where(hist >= max(8, int(max_count * 0.55)))[0]
    if candidate_bins.size == 0:
        candidate_bins = np.array([int(np.argmax(hist))])

    # For grasping visible food, prefer the nearest strong foreground depth
    # cluster over a farther background/table cluster.
    chosen_bin = int(candidate_bins[0])
    low = edges[chosen_bin]
    high = edges[chosen_bin + 1]
    keep = (z_values >= low) & (z_values < high)
    if keep.sum() < 8:
        chosen_bin = int(np.argmax(hist))
        low = edges[chosen_bin]
        high = edges[chosen_bin + 1]
        keep = (z_values >= low) & (z_values < high)

    return xs[keep], ys[keep], z_values[keep]


def select_connected_depth_component(xs, ys, z_values, height, width, intrinsics, prior_xyz=None):
    if z_values.size < 8:
        return xs, ys, z_values

    component_mask = np.zeros((height, width), dtype=np.uint8)
    component_mask[ys, xs] = 1
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(component_mask, 8)
    if num_labels <= 1:
        return xs, ys, z_values

    chosen_label = None
    if prior_xyz is not None:
        prior_pixel = project_point(intrinsics, prior_xyz, width, height)
        if prior_pixel is not None:
            prior_u, prior_v = prior_pixel
            if 0 <= prior_u < width and 0 <= prior_v < height:
                label = int(labels[prior_v, prior_u])
                if label > 0 and stats[label, cv2.CC_STAT_AREA] >= 8:
                    chosen_label = label

    if chosen_label is None:
        areas = stats[1:, cv2.CC_STAT_AREA]
        chosen_label = int(np.argmax(areas)) + 1

    keep = labels[ys, xs] == chosen_label
    if keep.sum() < 8:
        return xs, ys, z_values

    return xs[keep], ys[keep], z_values[keep]


def mask_pointcloud_center_xyz(mask: np.ndarray, depth_m: np.ndarray, intrinsics, prior_xyz=None):
    if depth_m is None or intrinsics is None:
        return None

    height, width = depth_m.shape[:2]
    mask = mask[:height, :width].astype(np.uint8)
    if mask.sum() < 8:
        return None

    # Keep the central, reliable part of the segmentation. Object boundaries
    # often contain mixed foreground/background depth and are bad for grasping.
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if distance.max() > 0:
        inner = (distance >= max(1.0, float(distance.max()) * 0.18)).astype(np.uint8)
        if inner.sum() >= 8:
            mask = inner

    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > MIN_VALID_DEPTH_M) & (depth_m < MAX_VALID_DEPTH_M)
    if valid.sum() < 8:
        return None

    ys, xs = np.where(valid)
    z_values = depth_m[ys, xs].astype(np.float64)
    prior_z = None if prior_xyz is None else float(prior_xyz[2])
    xs, ys, z_values = select_depth_cluster(xs, ys, z_values, prior_z=prior_z)
    if z_values.size < 8:
        return None

    xs, ys, z_values = select_connected_depth_component(xs, ys, z_values, height, width, intrinsics, prior_xyz)
    if z_values.size < 8:
        return None

    median_z = np.median(z_values)
    mad_z = np.median(np.abs(z_values - median_z))
    mad_z = max(float(mad_z), 1e-6)
    z_threshold = max(0.025, 3.0 * 1.4826 * mad_z)
    keep = np.abs(z_values - median_z) <= z_threshold
    if keep.sum() >= 8:
        xs = xs[keep]
        ys = ys[keep]
        z_values = z_values[keep]

    z = float(np.median(z_values))
    u = int(round(np.median(xs)))
    v = int(round(np.median(ys)))
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    if not mask[v, u]:
        nearest = nearest_mask_pixel(mask.astype(bool), (u, v))
        if nearest is not None:
            u, v = nearest

    return (u, v), deproject_pixel(intrinsics, (u, v), z)


def robust_depth_coordinate(mask: np.ndarray, depth_m: np.ndarray, intrinsics):
    if depth_m is None or intrinsics is None:
        return None

    height, width = depth_m.shape[:2]
    mask = mask[:height, :width].astype(bool)
    valid_depth = np.isfinite(depth_m) & (depth_m > MIN_VALID_DEPTH_M) & (depth_m < MAX_VALID_DEPTH_M)
    valid_mask = mask & valid_depth
    if valid_mask.sum() < 8:
        return None

    mask_u8 = valid_mask.astype(np.uint8)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if distance.max() > 0:
        inner = (distance >= max(2.0, float(distance.max()) * 0.30)) & valid_mask
        if inner.sum() >= 8:
            valid_mask = inner

    ys, xs = np.where(valid_mask)
    depths = depth_m[ys, xs].astype(np.float64)
    median_depth = np.median(depths)
    mad = np.median(np.abs(depths - median_depth))
    mad = max(float(mad), 1e-6)
    threshold = max(0.02, 3.5 * 1.4826 * mad)
    keep = np.abs(depths - median_depth) <= threshold
    if keep.sum() >= 8:
        xs = xs[keep]
        ys = ys[keep]
        depths = depths[keep]

    # Pick a representative pixel inside the object and use the robust local
    # depth there. Manual RealSense deprojection keeps the coordinate formula
    # explicit and easy to audit:
    # X = (u - ppx) / fx * Z, Y = (v - ppy) / fy * Z, Z = depth.
    u = int(np.round(np.median(xs)))
    v = int(np.round(np.median(ys)))
    if not mask[v, u]:
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        nearest = np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2)
        u = int(xs[nearest])
        v = int(ys[nearest])

    return local_depth_xyz(mask, depth_m, intrinsics, (u, v))


class StableObjectCoordinateEstimator:
    def __init__(
        self,
        iou_threshold=0.25,
        alpha=0.25,
        max_jump=0.12,
        lock_center=True,
        depth_mode="pointcloud_median",
        max_bbox_area_ratio=0.70,
        history_size=10,
        min_history=3,
    ):
        self.iou_threshold = iou_threshold
        self.alpha = alpha
        self.max_jump = max_jump
        self.lock_center = lock_center
        self.depth_mode = depth_mode
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.history_size = history_size
        self.min_history = min_history
        self.next_id = 1
        self.tracks = {}

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, mask_dict, depth_m, intrinsics):
        current = []
        for _sam_id, obj_info in mask_dict.labels.items():
            if obj_info.mask is None:
                continue
            mask = obj_info.mask.detach().cpu().numpy().astype(bool)
            if mask.sum() == 0:
                continue
            box = mask_box(mask)
            current.append(
                {
                    "class_name": obj_info.class_name or "object",
                    "mask": mask,
                    "box": box,
                }
            )

        assigned_track_ids = set()
        for obj in current:
            best_track_id = None
            best_iou = 0.0
            best_center_distance = float("inf")
            obj_center = box_center(obj["box"])
            for track_id, track in self.tracks.items():
                if track_id in assigned_track_ids:
                    continue
                if track["class_name"] != obj["class_name"]:
                    continue
                iou = mask_iou(obj["mask"], track["mask"])
                track_center = box_center(track["box"])
                center_distance = float("inf")
                if obj_center is not None and track_center is not None:
                    center_distance = float(np.linalg.norm(obj_center - track_center))
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
                    best_center_distance = center_distance

            if best_track_id is None or (best_iou < self.iou_threshold and best_center_distance > 90.0):
                best_track_id = self.next_id
                self.next_id += 1

            assigned_track_ids.add(best_track_id)

            pixel = None
            xyz = None
            previous_track = self.tracks.get(best_track_id, {})
            previous_xyz = previous_track.get("xyz")
            previous_valid_xyz = previous_track.get("last_valid_xyz", previous_xyz)
            xyz_history = previous_track.get("xyz_history")
            if xyz_history is None:
                xyz_history = deque(maxlen=self.history_size)
            elif xyz_history.maxlen != self.history_size:
                xyz_history = deque(xyz_history, maxlen=self.history_size)
            locked_relative = previous_track.get("locked_relative")
            height, width = obj["mask"].shape[:2]
            x1, y1, x2, y2 = obj["box"]
            bbox_area_ratio = ((x2 - x1 + 1) * (y2 - y1 + 1)) / max(width * height, 1)
            raw_depth = mask_median_raw_depth(obj["mask"], depth_m)
            too_close_by_depth = raw_depth is not None and raw_depth < MIN_VALID_DEPTH_M
            too_close = bbox_area_ratio >= self.max_bbox_area_ratio or too_close_by_depth

            if too_close:
                coord = None
            elif self.depth_mode == "pointcloud_median":
                coord = mask_pointcloud_center_xyz(obj["mask"], depth_m, intrinsics, prior_xyz=previous_valid_xyz)
            elif self.depth_mode == "mask_mean":
                coord = mask_average_depth_xyz(obj["mask"], depth_m, intrinsics)
            elif self.lock_center and locked_relative is not None:
                target_pixel = box_relative_to_pixel(locked_relative, obj["box"])
                coord = local_depth_xyz(obj["mask"], depth_m, intrinsics, target_pixel)
                if coord is None:
                    coord = robust_depth_coordinate(obj["mask"], depth_m, intrinsics)
            else:
                coord = robust_depth_coordinate(obj["mask"], depth_m, intrinsics)
                if self.lock_center and coord is not None:
                    locked_relative = pixel_to_box_relative(coord[0], obj["box"])

            if coord is not None:
                pixel, measured_xyz = coord
                if self.lock_center and locked_relative is None:
                    locked_relative = pixel_to_box_relative(pixel, obj["box"])
                previous = previous_valid_xyz
                if previous is None:
                    xyz = measured_xyz
                else:
                    delta = measured_xyz - previous
                    jump = float(np.linalg.norm(delta))
                    if jump > self.max_jump:
                        measured_xyz = previous + delta * (self.max_jump / jump)
                xyz_history.append(measured_xyz)
                history_values = np.stack(list(xyz_history), axis=0)
                median_xyz = np.median(history_values, axis=0)
                if previous is not None and len(xyz_history) >= self.min_history:
                    xyz = self.alpha * median_xyz + (1.0 - self.alpha) * previous
                else:
                    xyz = median_xyz

            self.tracks[best_track_id] = {
                "class_name": obj["class_name"],
                "mask": obj["mask"],
                "box": obj["box"],
                "pixel": pixel,
                "locked_relative": locked_relative,
                "xyz": xyz,
                "last_valid_xyz": xyz if xyz is not None else previous_valid_xyz,
                "xyz_history": xyz_history,
                "too_close": too_close,
                "missing": 0,
            }

        for track_id in list(self.tracks.keys()):
            if track_id not in assigned_track_ids:
                self.tracks[track_id]["missing"] += 1
                if self.tracks[track_id]["missing"] > 15:
                    del self.tracks[track_id]
            else:
                self.tracks[track_id]["missing"] = 0

        return [
            {"id": track_id, **track}
            for track_id, track in sorted(self.tracks.items())
            if track["missing"] == 0
        ]


def draw_stable_objects(image_rgb: np.ndarray, objects):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    palette = [
        (255, 128, 0),
        (0, 180, 255),
        (120, 90, 255),
        (80, 220, 80),
        (255, 80, 180),
        (0, 220, 220),
    ]

    for obj in objects:
        track_id = obj["id"]
        color = palette[(track_id - 1) % len(palette)]
        mask = obj["mask"]
        box = obj["box"]
        if box is None:
            continue

        overlay = image_bgr.copy()
        overlay[mask] = color
        image_bgr = cv2.addWeighted(overlay, 0.35, image_bgr, 0.65, 0)

        x1, y1, x2, y2 = box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)

        if obj["xyz"] is not None and obj["pixel"] is not None:
            u, v = obj["pixel"]
            x_m, y_m, z_m = obj["xyz"]
            cv2.drawMarker(
                image_bgr,
                (u, v),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.circle(image_bgr, (u, v), 4, (0, 0, 255), -1)
            label = f'{track_id}: {obj["class_name"]} ({x_m:.2f}, {y_m:.2f}, {z_m:.2f}) m'
        elif obj.get("too_close"):
            label = f'{track_id}: {obj["class_name"]} (too close)'
        else:
            label = f'{track_id}: {obj["class_name"]} (no depth)'

        text_origin = (x1, max(y1 - 8, 18))
        cv2.putText(
            image_bgr,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class GroundedSAM2CameraApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("Grounded SAM 2 Camera Demo")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.prompt_lock = threading.Lock()
        self.current_prompt = normalize_prompt(args.prompt)
        self.pending_prompt = None
        self.worker = None
        self.photo = None
        self.latest_frame_rgb = None
        self.coordinate_estimator = StableObjectCoordinateEstimator(
            alpha=args.xyz_alpha,
            max_jump=args.xyz_max_jump,
            lock_center=not args.no_lock_center,
            depth_mode=args.depth_mode,
            max_bbox_area_ratio=args.max_bbox_area_ratio,
            history_size=args.xyz_history_size,
            min_history=args.xyz_min_history,
        )

        self._build_ui()
        self._start_worker()
        self._poll_frame()

    def _build_ui(self):
        self.video_label = ttk.Label(self.root, anchor="center")
        self.video_label.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.video_label.bind("<Configure>", lambda _event: self._render_latest_frame())

        ttk.Label(self.root, text="Prompt").grid(row=1, column=0, padx=(10, 4), pady=8)
        self.prompt_var = tk.StringVar(value=self.current_prompt)
        self.prompt_entry = ttk.Entry(self.root, textvariable=self.prompt_var, width=32)
        self.prompt_entry.grid(row=1, column=1, padx=4, pady=8, sticky="ew")
        self.prompt_entry.bind("<Return>", lambda _event: self.update_prompt())

        self.update_button = ttk.Button(self.root, text="Update", command=self.update_prompt)
        self.update_button.grid(row=1, column=2, padx=4, pady=8)

        self.quit_button = ttk.Button(self.root, text="Quit", command=self.close)
        self.quit_button.grid(row=1, column=3, padx=(4, 10), pady=8)

        self.status_var = tk.StringVar(value="Loading models...")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=4, padx=10, pady=(0, 10), sticky="w")

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _start_worker(self):
        self.worker = threading.Thread(target=self._run_camera_loop, daemon=True)
        self.worker.start()

    def update_prompt(self):
        prompt = normalize_prompt(self.prompt_var.get())
        self.prompt_var.set(prompt)
        with self.prompt_lock:
            self.pending_prompt = prompt
        self.status_var.set(f"Prompt update queued: {prompt}")

    def _run_camera_loop(self):
        device = "cuda"
        tracker = IncrementalObjectTracker(
            grounding_model_id="IDEA-Research/grounding-dino-tiny",
            sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
            sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
            device=device,
            prompt_text=self.current_prompt,
            detection_interval=self.args.detection_interval,
            box_threshold=self.args.box_threshold,
            text_threshold=self.args.text_threshold,
        )
        tracker.set_prompt(self.current_prompt)

        cap = None
        camera = None
        camera_index = None
        use_realsense = self.args.source == "realsense"
        if use_realsense:
            try:
                camera = RealSenseColorDepthCamera(self.args.width, self.args.height, self.args.fps)
                camera.start()
                camera_index = "RealSense"
            except Exception as exc:
                self._set_status(f"RealSense failed: {exc}. Falling back to OpenCV camera.")
                use_realsense = False

        if not use_realsense:
            cap, camera_index = open_camera(self.args.camera)
            if cap is None:
                self._set_status("Cannot open camera. Try another --camera index.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

        self._set_status(f"Camera {camera_index} opened. Tracking: {self.current_prompt}")

        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                if use_realsense:
                    ret, frame_rgb, depth_m = camera.read()
                    intrinsics = camera.intrinsics
                else:
                    ret, frame_bgr = cap.read()
                    depth_m = None
                    intrinsics = None
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if ret else None

                if not ret:
                    self._set_status("Failed to capture frame.")
                    time.sleep(0.05)
                    continue

                with self.prompt_lock:
                    new_prompt = self.pending_prompt
                    self.pending_prompt = None
                if new_prompt:
                    self.current_prompt = new_prompt
                    tracker.set_prompt(new_prompt)
                    self.coordinate_estimator.reset()
                    self._set_status(f"Tracking: {new_prompt}")

                tracker.add_image(frame_rgb)
                objects = self.coordinate_estimator.update(
                    tracker.last_mask_dict,
                    depth_m,
                    intrinsics,
                )
                annotated_rgb = draw_stable_objects(frame_rgb, objects)

                self._put_frame(annotated_rgb)
                frame_idx += 1
                self._set_status(
                    f"Camera {camera_index} | prompt: {self.current_prompt} | frame: {frame_idx} | objects: {len(objects)}"
                )
        finally:
            if cap is not None:
                cap.release()
            if camera is not None:
                camera.stop()

    def _put_frame(self, frame_rgb):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame_rgb)

    def _poll_frame(self):
        try:
            self.latest_frame_rgb = self.frame_queue.get_nowait()
        except queue.Empty:
            if not self.stop_event.is_set():
                self.root.after(30, self._poll_frame)
            return

        self._render_latest_frame()

        if not self.stop_event.is_set():
            self.root.after(30, self._poll_frame)

    def _render_latest_frame(self):
        if self.latest_frame_rgb is None:
            return

        max_width = max(self.video_label.winfo_width(), 1)
        max_height = max(self.video_label.winfo_height(), 1)
        if max_width <= 1 or max_height <= 1:
            max_width = self.args.display_width
            max_height = self.args.display_height

        image = Image.fromarray(self.latest_frame_rgb)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo)

    def _set_status(self, text):
        self.root.after(0, self.status_var.set, text)

    def close(self):
        self.stop_event.set()
        self.root.after(100, self.root.destroy)


def parse_args():
    parser = argparse.ArgumentParser(description="Grounded SAM 2 camera UI demo.")
    parser.add_argument("--camera", type=int, default=4, help="Preferred OpenCV camera index.")
    parser.add_argument("--source", choices=["realsense", "opencv"], default="realsense", help="Camera backend.")
    parser.add_argument("--prompt", type=str, default="person.", help="Initial text prompt.")
    parser.add_argument("--detection-interval", type=int, default=20, help="Run Grounding DINO every N frames.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box confidence threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.30, help="GroundingDINO text confidence threshold.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested RealSense FPS.")
    parser.add_argument("--xyz-alpha", type=float, default=0.2, help="Temporal smoothing factor for XYZ coordinates.")
    parser.add_argument("--xyz-max-jump", type=float, default=0.08, help="Maximum accepted per-frame XYZ jump before clamping, in meters.")
    parser.add_argument("--xyz-history-size", type=int, default=10, help="Number of recent XYZ samples used for median smoothing.")
    parser.add_argument("--xyz-min-history", type=int, default=3, help="Minimum XYZ samples before applying extra temporal smoothing.")
    parser.add_argument("--no-lock-center", action="store_true", help="Disable per-object locked center point for XYZ sampling.")
    parser.add_argument("--depth-mode", choices=["pointcloud_median", "mask_mean", "locked_point"], default="pointcloud_median", help="Depth strategy for object coordinates.")
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.70, help="Hide XYZ when an object's bbox covers this fraction of the image.")
    parser.add_argument("--display-width", type=int, default=960, help="Max display width.")
    parser.add_argument("--display-height", type=int, default=720, help="Max display height.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    GroundedSAM2CameraApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
