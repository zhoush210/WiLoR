from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


HAND_SUFFIXES_OPENPOSE = [
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

HAND_COLORS_BGR = {
    "left": (185, 88, 240),
    "right": (245, 150, 60),
}
LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
METERS_TO_MILLIMETERS = 1000.0


def egodex_joint_names(hand: str) -> List[str]:
    if hand not in {"left", "right"}:
        raise ValueError(f"Unsupported hand '{hand}'. Expected 'left' or 'right'.")
    return [f"{hand}{suffix}" for suffix in HAND_SUFFIXES_OPENPOSE]


def selected_hands(hand_arg: str) -> List[str]:
    return ["left", "right"] if hand_arg == "both" else [hand_arg]


def resolve_episode_paths(dataset_path: str | Path, episode: str | int) -> Tuple[Path, Path]:
    dataset_path = Path(dataset_path)
    episode_name = Path(str(episode)).stem
    video_path = dataset_path / f"{episode_name}.mp4"
    gt_path = dataset_path / f"{episode_name}.hdf5"
    missing = [str(path) for path in (video_path, gt_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing EgoDex episode file(s): " + ", ".join(missing))
    return video_path, gt_path


def default_output_dir(dataset_path: str | Path, episode: str | int) -> Path:
    dataset_name = Path(dataset_path).resolve().name or "egodex"
    episode_name = Path(str(episode)).stem
    return REPO_ROOT / "results" / "egodex" / dataset_name / episode_name


def video_info(video_path: str | Path) -> Dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def _jsonify_attr(value):
    if isinstance(value, np.ndarray):
        return [_jsonify_attr(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_egodex_gt(gt_path: str | Path, confidence_thresh: float = 0.0) -> Dict:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "scripts/test_egodex.py requires h5py to read EgoDex .hdf5 files. "
            "Install it in the active wilor environment with: python -m pip install h5py"
        ) from exc

    gt_path = Path(gt_path)
    with h5py.File(gt_path, "r") as h5:
        intrinsic = np.asarray(h5["camera/intrinsic"][()], dtype=np.float32)
        camera_transforms = np.asarray(h5["transforms/camera"][()], dtype=np.float32)
        attrs = {key: _jsonify_attr(value) for key, value in h5.attrs.items()}

        hands = {}
        for hand in ("left", "right"):
            joints = []
            confidences = []
            names = egodex_joint_names(hand)
            for name in names:
                transform_key = f"transforms/{name}"
                confidence_key = f"confidences/{name}"
                if transform_key not in h5:
                    raise KeyError(f"Missing EgoDex transform '{transform_key}' in {gt_path}")
                if confidence_key not in h5:
                    raise KeyError(f"Missing EgoDex confidence '{confidence_key}' in {gt_path}")
                joints.append(np.asarray(h5[transform_key][:, :3, 3], dtype=np.float32))
                confidences.append(np.asarray(h5[confidence_key][()], dtype=np.float32))

            joints_np = np.stack(joints, axis=1)
            confidences_np = np.stack(confidences, axis=1)
            valid = np.isfinite(joints_np).all(axis=(1, 2))
            valid &= np.all(confidences_np >= float(confidence_thresh), axis=1)
            hands[hand] = {
                "joints_world": joints_np,
                "confidence": confidences_np,
                "valid": valid,
                "joint_names": names,
            }

    for hand, hand_data in hands.items():
        hand_data["joints"] = world_to_camera_joints(hand_data["joints_world"], camera_transforms)

    return {
        "intrinsic": intrinsic,
        "camera_transforms": camera_transforms,
        "attrs": attrs,
        "hands": hands,
    }


def world_to_camera_joints(joints_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    joints_world = np.asarray(joints_world, dtype=np.float32)
    camera_transforms = np.asarray(camera_transforms, dtype=np.float32)
    length = min(joints_world.shape[0], camera_transforms.shape[0])
    joints_world = joints_world[:length]
    camera_transforms = camera_transforms[:length]
    r_c2w = camera_transforms[:, :3, :3]
    t_c2w = camera_transforms[:, :3, 3]
    r_w2c = np.transpose(r_c2w, (0, 2, 1))
    centered = joints_world - t_c2w[:, None, :]
    return np.einsum("tij,tnj->tni", r_w2c, centered).astype(np.float32)


def project_camera_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    z = points[..., 2]
    valid = np.isfinite(points).all(axis=-1) & (z > 1e-6)
    safe_z = np.where(valid, z, 1.0)
    normalized = points / safe_z[..., None]
    projected = np.einsum("ij,...j->...i", intrinsic, normalized)[..., :2]
    return projected.astype(np.float32), valid


def compute_mpjpe_errors(pred_joints: np.ndarray, gt_joints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_joints, dtype=np.float32)
    gt = np.asarray(gt_joints, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    length = min(pred.shape[0], gt.shape[0], valid.shape[0])
    if length == 0:
        return np.empty((0,), dtype=np.float32)
    pred = pred[:length]
    gt = gt[:length]
    valid = valid[:length]
    joint_valid = valid[:, None] & np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    distances = np.linalg.norm(pred - gt, axis=-1)
    return distances[joint_valid].astype(np.float32)


def compute_accel_errors_m_per_s2(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    valid: np.ndarray,
    fps: float,
) -> np.ndarray:
    pred = np.asarray(pred_joints, dtype=np.float32)
    gt = np.asarray(gt_joints, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    length = min(pred.shape[0], gt.shape[0], valid.shape[0])
    if length < 3:
        return np.empty((0,), dtype=np.float32)
    pred = pred[:length]
    gt = gt[:length]
    valid = valid[:length]
    pred_accel = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    gt_accel = gt[2:] - 2.0 * gt[1:-1] + gt[:-2]
    window_valid = valid[2:] & valid[1:-1] & valid[:-2]
    joint_valid = window_valid[:, None] & np.isfinite(pred_accel).all(axis=-1) & np.isfinite(gt_accel).all(axis=-1)
    errors = np.linalg.norm(pred_accel - gt_accel, axis=-1) * (float(fps) ** 2)
    return errors[joint_valid].astype(np.float32)


def summarize_values(values: Iterable[float]) -> Dict[str, Optional[float] | int]:
    values_np = np.asarray(list(values), dtype=np.float64).reshape(-1)
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values_np.size),
        "mean": float(values_np.mean()),
        "median": float(np.median(values_np)),
        "std": float(values_np.std()),
        "min": float(values_np.min()),
        "max": float(values_np.max()),
    }


def evaluate_predictions(pred_hands: Dict, gt: Dict, hands: Sequence[str], fps: float) -> Dict:
    per_hand = {}
    combined = {"mpjpe_mm": [], "accel_m_per_s2": []}
    for hand in hands:
        pred = pred_hands[hand]
        target = gt["hands"][hand]
        frame_indices = pred.get("frame_indices", np.arange(pred["joints"].shape[0], dtype=np.int32))
        in_range = frame_indices < target["joints"].shape[0]
        pred_joints = pred["joints"][in_range]
        pred_valid = pred["valid"][in_range]
        gt_joints = target["joints"][frame_indices[in_range]]
        gt_valid = target["valid"][frame_indices[in_range]]
        valid = pred_valid & gt_valid
        length = pred_joints.shape[0]
        mpjpe_mm = compute_mpjpe_errors(pred_joints, gt_joints, valid) * METERS_TO_MILLIMETERS
        accel_m_per_s2 = compute_accel_errors_m_per_s2(
            pred_joints,
            gt_joints,
            valid,
            fps=fps,
        )
        per_hand[hand] = {
            "valid_frames": int(valid.sum()),
            "compared_frames": int(length),
            "metrics": {
                "mpjpe_mm": summarize_values(mpjpe_mm),
                "accel_m_per_s2": summarize_values(accel_m_per_s2),
            },
        }
        combined["mpjpe_mm"].append(mpjpe_mm)
        combined["accel_m_per_s2"].append(accel_m_per_s2)

    overall = {}
    for name, arrays in combined.items():
        arrays = [array for array in arrays if array.size > 0]
        overall[name] = summarize_values(np.concatenate(arrays) if arrays else np.empty((0,), dtype=np.float32))
    return {"hands": per_hand, "overall": overall}


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def build_joint_curve_data(pred_hands: Dict, gt: Dict, hand: str) -> Optional[Dict]:
    pred = pred_hands[hand]
    target = gt["hands"][hand]
    frame_indices = pred.get("frame_indices", np.arange(pred["joints"].shape[0], dtype=np.int32))
    in_range = frame_indices < target["joints"].shape[0]
    pred_joints = pred["joints"][in_range]
    pred_valid = pred["valid"][in_range]
    gt_joints = target["joints"][frame_indices[in_range]]
    gt_valid = target["valid"][frame_indices[in_range]]
    source_frames = frame_indices[in_range]
    valid = pred_valid & gt_valid
    if int(valid.sum()) == 0:
        return None
    frames = source_frames[valid]
    pred_values = pred_joints[valid]
    gt_values = gt_joints[valid]
    return {
        "hand": hand,
        "frames": frames,
        "pred": pred_values,
        "gt": gt_values,
        "error_mm": np.linalg.norm(pred_values - gt_values, axis=-1) * METERS_TO_MILLIMETERS,
        "joint_names": target.get("joint_names", egodex_joint_names(hand)),
    }


def save_joint_curve_csv(curve_data: Dict, csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame",
            "joint_index",
            "joint_name",
            "pred_x_m",
            "pred_y_m",
            "pred_z_m",
            "gt_x_m",
            "gt_y_m",
            "gt_z_m",
            "error_mm",
        ])
        for frame_i, frame in enumerate(curve_data["frames"]):
            for joint_i, joint_name in enumerate(curve_data["joint_names"]):
                pred_xyz = curve_data["pred"][frame_i, joint_i]
                gt_xyz = curve_data["gt"][frame_i, joint_i]
                writer.writerow([
                    int(frame),
                    joint_i,
                    joint_name,
                    float(pred_xyz[0]),
                    float(pred_xyz[1]),
                    float(pred_xyz[2]),
                    float(gt_xyz[0]),
                    float(gt_xyz[1]),
                    float(gt_xyz[2]),
                    float(curve_data["error_mm"][frame_i, joint_i]),
                ])


def plot_joint_xyz_curves(curve_data: Dict, output_path: str | Path) -> None:
    plt = _setup_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(7, 3, figsize=(18, 22), sharex=True)
    axes = axes.reshape(-1)
    axis_colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
    for joint_i, ax in enumerate(axes[:21]):
        joint_name = curve_data["joint_names"][joint_i]
        for dim_i, dim_name in enumerate(("x", "y", "z")):
            color = axis_colors[dim_name]
            ax.plot(
                curve_data["frames"],
                curve_data["gt"][:, joint_i, dim_i],
                linestyle="--",
                linewidth=1.0,
                color=color,
                label=f"GT {dim_name}" if joint_i == 0 else None,
            )
            ax.plot(
                curve_data["frames"],
                curve_data["pred"][:, joint_i, dim_i],
                linestyle="-",
                linewidth=1.0,
                color=color,
                label=f"Pred {dim_name}" if joint_i == 0 else None,
            )
        ax.set_title(f"{joint_i:02d} {joint_name}", fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=7)
    for ax in axes[-3:]:
        ax.set_xlabel("Frame")
    fig.suptitle(f"{curve_data['hand']} hand predicted vs GT joint coordinates (camera frame, meters)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_joint_error_curves(curve_data: Dict, output_path: str | Path) -> None:
    plt = _setup_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 21))
    for joint_i, joint_name in enumerate(curve_data["joint_names"]):
        ax.plot(
            curve_data["frames"],
            curve_data["error_mm"][:, joint_i],
            label=f"{joint_i:02d} {joint_name}",
            linewidth=1.4,
            color=colors[joint_i % len(colors)],
        )
    ax.set_title(f"{curve_data['hand']} hand per-joint 3D error")
    ax.set_xlabel("Frame")
    ax.set_ylabel("3D position error (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_joint_curve_outputs(pred_hands: Dict, gt: Dict, hands: Sequence[str], output_dir: str | Path) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for hand in hands:
        curve_data = build_joint_curve_data(pred_hands, gt, hand)
        if curve_data is None:
            outputs[hand] = {"valid": False}
            continue
        csv_path = output_dir / f"{hand}_joint_curves.csv"
        xyz_plot_path = output_dir / f"{hand}_joint_xyz_curves.png"
        error_plot_path = output_dir / f"{hand}_joint_error_curves.png"
        save_joint_curve_csv(curve_data, csv_path)
        plot_joint_xyz_curves(curve_data, xyz_plot_path)
        plot_joint_error_curves(curve_data, error_plot_path)
        outputs[hand] = {
            "valid": True,
            "csv": str(csv_path),
            "xyz_plot": str(xyz_plot_path),
            "error_plot": str(error_plot_path),
        }
    return outputs


def build_h264_transcode_command(
    input_path: str | Path,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


class CompatibleMP4Writer:
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_bin = shutil.which("ffmpeg")
        self.released = False
        self.encoded_tmp_path: Optional[Path] = None

        if self.ffmpeg_bin:
            raw_file = tempfile.NamedTemporaryFile(
                prefix=f".{self.output_path.stem}.",
                suffix=".raw.mp4",
                dir=str(self.output_path.parent),
                delete=False,
            )
            raw_file.close()
            self.raw_path = Path(raw_file.name)
            writer_path = self.raw_path
        else:
            self.raw_path = self.output_path
            writer_path = self.output_path
            print("WARNING: ffmpeg not found; writing OpenCV mp4v video, which may not play in VSCode.")

        self.writer = cv2.VideoWriter(
            str(writer_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {writer_path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.writer.release()

        if not self.ffmpeg_bin:
            return

        encoded_file = tempfile.NamedTemporaryFile(
            prefix=f".{self.output_path.stem}.",
            suffix=".h264.mp4",
            dir=str(self.output_path.parent),
            delete=False,
        )
        encoded_file.close()
        self.encoded_tmp_path = Path(encoded_file.name)

        command = build_h264_transcode_command(self.raw_path, self.encoded_tmp_path, self.ffmpeg_bin)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            try:
                self.encoded_tmp_path.unlink(missing_ok=True)
            finally:
                raise RuntimeError(
                    "ffmpeg failed to transcode video to VSCode-compatible H.264 MP4:\n"
                    + result.stderr.strip()
                )

        os.replace(self.encoded_tmp_path, self.output_path)
        self.raw_path.unlink(missing_ok=True)


def open_video_writer(output_path: str | Path, fps: float, width: int, height: int) -> CompatibleMP4Writer:
    output_path = Path(output_path)
    return CompatibleMP4Writer(output_path, fps, width, height)


def draw_skeleton(
    frame_bgr: np.ndarray,
    joints_camera: np.ndarray,
    intrinsic: np.ndarray,
    color_bgr: Tuple[int, int, int],
    label: Optional[str] = None,
) -> None:
    uv, valid = project_camera_points(joints_camera, intrinsic)
    height, width = frame_bgr.shape[:2]
    in_frame = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    points = np.round(uv).astype(np.int32)
    for start, end in HAND_EDGES:
        if in_frame[start] and in_frame[end]:
            cv2.line(frame_bgr, tuple(points[start]), tuple(points[end]), color_bgr, 2, cv2.LINE_AA)
    for joint_i, point in enumerate(points):
        if in_frame[joint_i]:
            radius = 4 if joint_i == 0 else 3
            cv2.circle(frame_bgr, tuple(point), radius, color_bgr, -1, cv2.LINE_AA)
    if label:
        valid_points = points[in_frame]
        if valid_points.size:
            anchor = tuple(valid_points[0])
            cv2.putText(frame_bgr, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)


def gt_skeleton_label(hand: str, show_labels: bool = False) -> Optional[str]:
    return f"GT {hand}" if show_labels else None


def render_gt_skeleton_video(
    video_path: str | Path,
    gt: Dict,
    hands: Sequence[str],
    output_path: str | Path,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    show_labels: bool = False,
) -> Dict:
    info = video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    writer = open_video_writer(output_path, info["fps"] / frame_stride, info["width"], info["height"])
    frame_idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
            if max_frames is not None and written >= max_frames:
                break
            for hand in hands:
                hand_data = gt["hands"][hand]
                if frame_idx < hand_data["joints"].shape[0] and hand_data["valid"][frame_idx]:
                    draw_skeleton(
                        frame,
                        hand_data["joints"][frame_idx],
                        gt["intrinsic"],
                        HAND_COLORS_BGR[hand],
                        gt_skeleton_label(hand, show_labels=show_labels),
                    )
            writer.write(frame)
            frame_idx += 1
            written += 1
    finally:
        cap.release()
        writer.release()
    return {"path": str(output_path), "frames": written, "fps": info["fps"] / frame_stride}


def _load_wilor_runtime(args):
    import torch
    from torch.utils.data import DataLoader
    from ultralytics import YOLO

    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.utils.renderer import Renderer, cam_crop_to_full

    model, model_cfg = load_wilor(checkpoint_path=args.checkpoint, cfg_path=args.cfg)
    if args.fast:
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone.skip_blocks = True
        if args.compile_backbone:
            model.backbone = torch.compile(model.backbone)

    detector = YOLO(args.detector)
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    model = model.to(device)
    detector = detector.to(device)
    model.eval()
    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "ViTDetDataset": ViTDetDataset,
        "recursive_to": recursive_to,
        "cam_crop_to_full": cam_crop_to_full,
        "model": model,
        "model_cfg": model_cfg,
        "detector": detector,
        "renderer": renderer,
        "device": device,
    }


def _detections_to_arrays(detections, conf_thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if detections.boxes is None or detections.boxes.data is None:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    data = detections.boxes.data.detach().cpu().numpy()
    if data.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    boxes = []
    right = []
    scores = []
    for row in data:
        score = float(row[4]) if row.shape[0] > 4 else 1.0
        if score < conf_thresh:
            continue
        boxes.append(row[:4].astype(np.float32))
        scores.append(score)
        right.append(float(round(float(row[5]))) if row.shape[0] > 5 else 1.0)
    if not boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.stack(boxes).astype(np.float32), np.asarray(right, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def _empty_prediction_store() -> Dict[str, Dict[str, List]]:
    return {
        hand: {"frame_indices": [], "joints": [], "valid": [], "boxes": [], "scores": []}
        for hand in ("left", "right")
    }


def _append_empty_frame(predictions: Dict[str, Dict[str, List]], frame_idx: int) -> None:
    for hand in ("left", "right"):
        predictions[hand]["frame_indices"].append(int(frame_idx))
        predictions[hand]["joints"].append(np.full((21, 3), np.nan, dtype=np.float32))
        predictions[hand]["valid"].append(False)
        predictions[hand]["boxes"].append(np.full((4,), np.nan, dtype=np.float32))
        predictions[hand]["scores"].append(np.nan)


def _set_frame_prediction(
    predictions: Dict[str, Dict[str, List]],
    hand: str,
    joints: np.ndarray,
    box: np.ndarray,
    score: float,
) -> None:
    frame_pos = len(predictions[hand]["joints"]) - 1
    current_score = predictions[hand]["scores"][frame_pos]
    if np.isfinite(current_score) and current_score >= score:
        return
    predictions[hand]["joints"][frame_pos] = joints.astype(np.float32)
    predictions[hand]["valid"][frame_pos] = True
    predictions[hand]["boxes"][frame_pos] = box.astype(np.float32)
    predictions[hand]["scores"][frame_pos] = float(score)


def _finalize_predictions(predictions: Dict[str, Dict[str, List]]) -> Dict:
    output = {}
    for hand, data in predictions.items():
        output[hand] = {
            "frame_indices": np.asarray(data["frame_indices"], dtype=np.int32),
            "joints": np.stack(data["joints"], axis=0).astype(np.float32),
            "valid": np.asarray(data["valid"], dtype=bool),
            "boxes": np.stack(data["boxes"], axis=0).astype(np.float32),
            "scores": np.asarray(data["scores"], dtype=np.float32),
            "joint_names": egodex_joint_names(hand),
        }
    return output


def run_wilor_on_video(
    args,
    video_path: str | Path,
    output_video_path: Optional[str | Path],
    focal_length: Optional[float] = None,
) -> Tuple[Dict, Dict]:
    runtime = _load_wilor_runtime(args)
    torch = runtime["torch"]
    DataLoader = runtime["DataLoader"]
    ViTDetDataset = runtime["ViTDetDataset"]
    recursive_to = runtime["recursive_to"]
    cam_crop_to_full = runtime["cam_crop_to_full"]
    model = runtime["model"]
    model_cfg = runtime["model_cfg"]
    detector = runtime["detector"]
    renderer = runtime["renderer"]
    device = runtime["device"]

    info = video_info(video_path)
    full_focal_length = float(focal_length) if focal_length is not None else (
        model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(info["width"], info["height"])
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = None
    if output_video_path is not None:
        writer = open_video_writer(output_video_path, info["fps"] / args.frame_stride, info["width"], info["height"])

    predictions = _empty_prediction_store()
    frame_idx = 0
    written = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride != 0:
                frame_idx += 1
                continue
            if args.max_frames is not None and processed >= args.max_frames:
                break

            _append_empty_frame(predictions, frame_idx)
            detections = detector(frame, conf=args.det_conf, verbose=False)[0]
            boxes, right, scores = _detections_to_arrays(detections, args.det_conf)
            render_verts = []
            render_cam_t = []
            render_right = []

            if boxes.shape[0] > 0:
                dataset = ViTDetDataset(
                    model_cfg,
                    frame,
                    boxes,
                    right,
                    rescale_factor=args.rescale_factor,
                    fp16=args.fast,
                )
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    with torch.no_grad():
                        out = model(batch)

                    multiplier = 2 * batch["right"] - 1
                    pred_cam = out["pred_cam"].clone()
                    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    pred_cam_t_full = cam_crop_to_full(
                        pred_cam,
                        box_center,
                        box_size,
                        img_size,
                        full_focal_length,
                    ).detach().cpu().numpy()

                    for item_idx in range(batch["img"].shape[0]):
                        person_id = int(batch["personid"][item_idx])
                        is_right = float(batch["right"][item_idx].detach().cpu().item())
                        hand = "right" if is_right >= 0.5 else "left"
                        side_multiplier = 2.0 * is_right - 1.0
                        verts = out["pred_vertices"][item_idx].detach().cpu().numpy()
                        joints = out["pred_keypoints_3d"][item_idx].detach().cpu().numpy()
                        verts[:, 0] = side_multiplier * verts[:, 0]
                        joints[:, 0] = side_multiplier * joints[:, 0]
                        cam_t = pred_cam_t_full[item_idx].astype(np.float32)
                        joints_camera = joints + cam_t[None, :]
                        _set_frame_prediction(
                            predictions,
                            hand,
                            joints_camera,
                            boxes[person_id],
                            float(scores[person_id]),
                        )
                        render_verts.append(verts)
                        render_cam_t.append(cam_t)
                        render_right.append(is_right)

            if writer is not None:
                output_frame = frame
                if render_verts:
                    cam_view = renderer.render_rgba_multiple(
                        render_verts,
                        cam_t=render_cam_t,
                        render_res=[info["width"], info["height"]],
                        is_right=render_right,
                        mesh_base_color=LIGHT_PURPLE,
                        scene_bg_color=(1, 1, 1),
                        focal_length=full_focal_length,
                    )
                    input_img = frame.astype(np.float32)[:, :, ::-1] / 255.0
                    input_rgba = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
                    overlay = input_rgba[:, :, :3] * (1.0 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                    output_frame = np.clip(overlay[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
                writer.write(output_frame)
                written += 1

            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed} frame(s)")
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return _finalize_predictions(predictions), {
        "frames": processed,
        "rendered_frames": written,
        "device": str(device),
        "focal_length": full_focal_length,
    }


def save_predictions_npz(pred_hands: Dict, gt: Dict, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pred_left_joints=pred_hands["left"]["joints"],
        pred_left_valid=pred_hands["left"]["valid"],
        pred_left_frame_indices=pred_hands["left"]["frame_indices"],
        pred_right_joints=pred_hands["right"]["joints"],
        pred_right_valid=pred_hands["right"]["valid"],
        pred_right_frame_indices=pred_hands["right"]["frame_indices"],
        gt_left_joints=gt["hands"]["left"]["joints"],
        gt_left_valid=gt["hands"]["left"]["valid"],
        gt_right_joints=gt["hands"]["right"]["joints"],
        gt_right_valid=gt["hands"]["right"]["valid"],
        camera_intrinsic=gt["intrinsic"],
        camera_transforms=gt["camera_transforms"],
    )
    return str(output_path)


def write_json(path: str | Path, data: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WiLoR on an EgoDex episode.")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--episode", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--confidence_thresh", type=float, default=0.0)
    parser.add_argument("--det_conf", type=float, default=0.3)
    parser.add_argument("--img_focal", type=float, default=None, help="Override EgoDex fx for WiLoR full-camera conversion.")
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--fast", action="store_true", help="Use FP16 and WiLoR backbone block skipping.")
    parser.add_argument("--compile_backbone", action="store_true", help="Also torch.compile the WiLoR backbone when --fast is set.")
    parser.add_argument("--cpu", action="store_true", help="Run WiLoR on CPU even when CUDA is available.")
    parser.add_argument("--skip_pred_video", action="store_true")
    parser.add_argument("--skip_gt_video", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    video_path, gt_path = resolve_episode_paths(args.dataset_path, args.episode)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataset_path, args.episode)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = video_info(video_path)
    hands = selected_hands(args.hand)
    gt = load_egodex_gt(gt_path, confidence_thresh=args.confidence_thresh)
    img_focal = float(args.img_focal) if args.img_focal is not None else float(gt["intrinsic"][0, 0])

    pred_video_path = None if args.skip_pred_video else output_dir / "wilor_mano_overlay.mp4"
    pred_hands, run_info = run_wilor_on_video(args, video_path, pred_video_path, focal_length=img_focal)
    metrics = evaluate_predictions(pred_hands, gt, hands, fps=float(info["fps"]) / args.frame_stride)

    outputs = {
        "wilor_mano_video": str(pred_video_path) if pred_video_path is not None else None,
        "predictions_npz": save_predictions_npz(pred_hands, gt, output_dir / "wilor_egodex_predictions.npz"),
    }
    if args.skip_gt_video:
        outputs["gt_skeleton_video"] = None
    else:
        gt_video_path = output_dir / "egodex_gt_skeleton.mp4"
        outputs["gt_skeleton_video"] = render_gt_skeleton_video(
            video_path,
            gt,
            hands,
            gt_video_path,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )

    if args.skip_plots:
        plot_outputs = {"enabled": False}
    else:
        plot_dir = output_dir / "joint_curves"
        plot_outputs = {
            "enabled": True,
            "dir": str(plot_dir),
            "hands": save_joint_curve_outputs(pred_hands, gt, hands, plot_dir),
        }

    result = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "episode": str(Path(str(args.episode)).stem),
        "video_path": str(video_path),
        "gt_path": str(gt_path),
        "output_dir": str(output_dir),
        "video": info,
        "img_focal": img_focal,
        "frame_stride": args.frame_stride,
        "processed_frames": run_info["frames"],
        "runtime": run_info,
        "gt_attrs": gt["attrs"],
        "metrics": metrics,
        "outputs": outputs,
        "plots": plot_outputs,
    }
    output_json = Path(args.output_json) if args.output_json else output_dir / "egodex_wilor_eval.json"
    write_json(output_json, result)

    print(f"EgoDex episode {args.episode} WiLoR metrics:")
    for name, summary in metrics["overall"].items():
        mean = summary["mean"]
        mean_text = "nan" if mean is None else f"{mean:.6f}"
        print(f"  overall {name}: {mean_text}")
    for hand, hand_result in metrics["hands"].items():
        print(f"  {hand}: {hand_result['valid_frames']} valid frame(s)")
        for name, summary in hand_result["metrics"].items():
            mean = summary["mean"]
            mean_text = "nan" if mean is None else f"{mean:.6f}"
            print(f"    {name}: {mean_text}")
    print(f"Saved results to {output_json}")
    if pred_video_path is not None:
        print(f"Saved WiLoR MANO overlay video to {pred_video_path}")
    if outputs["gt_skeleton_video"]:
        print(f"Saved EgoDex GT skeleton video to {outputs['gt_skeleton_video']['path']}")
    if plot_outputs["enabled"]:
        print(f"Saved joint comparison plots to {plot_outputs['dir']}")


if __name__ == "__main__":
    main()
