from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from scripts import test_egodex as egodex_utils  # noqa: E402


CAMERA_FILES = {
    "stereo_left": "stereo_left.mp4",
    "stereo_right": "stereo_right.mp4",
    "fisheye_cam0": "fisheye_cam0.mp4",
    "fisheye_cam1": "fisheye_cam1.mp4",
    "fisheye_cam2": "fisheye_cam2.mp4",
    "fisheye_cam3": "fisheye_cam3.mp4",
}


@dataclass(frozen=True)
class EpisodePaths:
    episode_dir: Path
    video_path: Path
    annotation_path: Path


def resolve_episode_paths(dataset_path: str | Path, episode: str | int, camera: str = "stereo_left") -> EpisodePaths:
    dataset_path = Path(dataset_path)
    episode_name = str(episode)
    if not episode_name.startswith("ep"):
        episode_name = f"ep{episode_name}"

    episode_dir = dataset_path / episode_name
    if not episode_dir.exists():
        raise FileNotFoundError(f"Missing episode folder: {episode_dir}")

    annotation_path = episode_dir / "annotation.hdf5"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    video_name = CAMERA_FILES.get(camera, camera)
    video_path = episode_dir / video_name
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    return EpisodePaths(episode_dir=episode_dir, video_path=video_path, annotation_path=annotation_path)


def intrinsic_vector_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] != 4:
        raise ValueError(f"Expected xperience-10m intrinsics [fx, fy, cx, cy], got shape {values.shape}.")
    fx, fy, cx, cy = values.tolist()
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def normalize_joints(joints: np.ndarray, name: str) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim == 2:
        if joints.shape == (21, 3):
            return joints[None, ...]
        if joints.shape == (3, 21):
            return joints.T[None, ...]
        if joints.shape[1] == 63:
            return joints.reshape(-1, 21, 3)
        raise ValueError(f"Unsupported {name} shape {joints.shape}")
    if joints.ndim != 3:
        raise ValueError(f"Unsupported {name} shape {joints.shape}")
    if joints.shape[-2:] == (21, 3):
        return joints
    if joints.shape[-2:] == (3, 21):
        return np.swapaxes(joints, -1, -2)
    if joints.shape[-1] == 63:
        return joints.reshape(joints.shape[0], 21, 3)
    raise ValueError(f"Unsupported {name} shape {joints.shape}")


def joint_valid_mask(joints: np.ndarray, confidence: Optional[np.ndarray] = None, confidence_thresh: float = 0.0) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    valid = np.isfinite(joints).all(axis=(1, 2))
    valid &= np.linalg.norm(joints.reshape(joints.shape[0], -1), axis=1) > 1e-8
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
        valid &= confidence[: joints.shape[0]] >= float(confidence_thresh)
    return valid


def compact_frame_ranges(frame_indices: np.ndarray) -> list[Dict[str, int]]:
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if frame_indices.size == 0:
        return []

    ranges = []
    start = prev = int(frame_indices[0])
    for frame_idx in frame_indices[1:]:
        frame_idx = int(frame_idx)
        if frame_idx == prev + 1:
            prev = frame_idx
            continue
        ranges.append({"start": start, "end": prev, "count": prev - start + 1})
        start = prev = frame_idx
    ranges.append({"start": start, "end": prev, "count": prev - start + 1})
    return ranges


def valid_frame_ranges(valid: np.ndarray) -> list[Dict[str, int]]:
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    return compact_frame_ranges(np.flatnonzero(valid))


def selected_source_frame_indices(frame_count: int, max_frames: Optional[int], frame_stride: int) -> np.ndarray:
    if frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    indices = np.arange(0, int(frame_count), int(frame_stride), dtype=np.int32)
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    return indices


def gt_visibility_summary(gt: Dict, hands: Sequence[str], frame_indices: Optional[np.ndarray] = None) -> Dict:
    summary = {}
    for hand in hands:
        valid = np.asarray(gt["hands"][hand]["valid"], dtype=bool).reshape(-1)
        if frame_indices is None:
            valid_indices = np.flatnonzero(valid)
        else:
            source_indices = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            source_indices = source_indices[source_indices < valid.shape[0]]
            valid_indices = source_indices[valid[source_indices]]
        summary[hand] = {
            "valid_frames": int(valid_indices.size),
            "first_valid_frame": int(valid_indices[0]) if valid_indices.size else None,
            "last_valid_frame": int(valid_indices[-1]) if valid_indices.size else None,
            "valid_ranges": compact_frame_ranges(valid_indices),
        }
    return summary


def load_xperience_gt(
    annotation_path: str | Path,
    confidence_thresh: float = 0.0,
    intrinsic_key: str = "calibration/cam01/K",
) -> Dict:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "scripts/test_xperience_10m.py requires h5py to read annotation.hdf5. "
            "Install it in the active wilor environment with: python -m pip install h5py"
        ) from exc

    annotation_path = Path(annotation_path)
    with h5py.File(annotation_path, "r") as h5:
        for key in (
            intrinsic_key,
            "hand_mocap/left_joints_3d",
            "hand_mocap/right_joints_3d",
        ):
            if key not in h5:
                raise KeyError(f"{annotation_path} is missing HDF5 dataset '{key}'")

        left_joints = normalize_joints(h5["hand_mocap/left_joints_3d"][()], "left_joints_3d")
        right_joints = normalize_joints(h5["hand_mocap/right_joints_3d"][()], "right_joints_3d")
        frame_count = min(left_joints.shape[0], right_joints.shape[0])
        left_joints = left_joints[:frame_count]
        right_joints = right_joints[:frame_count]

        left_conf = None
        right_conf = None
        if "hand_mocap/left_joints_3d_confidence" in h5:
            left_conf = np.asarray(h5["hand_mocap/left_joints_3d_confidence"][()], dtype=np.float32)[:frame_count]
        if "hand_mocap/right_joints_3d_confidence" in h5:
            right_conf = np.asarray(h5["hand_mocap/right_joints_3d_confidence"][()], dtype=np.float32)[:frame_count]

        intrinsic = intrinsic_vector_to_matrix(h5[intrinsic_key][()])
        attrs = {key: egodex_utils._jsonify_attr(value) for key, value in h5.attrs.items()}

    joint_names = egodex_utils.egodex_joint_names("right")
    return {
        "intrinsic": intrinsic,
        "camera_transforms": None,
        "attrs": attrs,
        "hands": {
            "left": {
                "joints": left_joints,
                "confidence": np.ones(frame_count, dtype=np.float32) if left_conf is None else left_conf,
                "valid": joint_valid_mask(left_joints, left_conf, confidence_thresh),
                "joint_names": joint_names,
            },
            "right": {
                "joints": right_joints,
                "confidence": np.ones(frame_count, dtype=np.float32) if right_conf is None else right_conf,
                "valid": joint_valid_mask(right_joints, right_conf, confidence_thresh),
                "joint_names": joint_names,
            },
        },
    }


def default_output_dir(dataset_path: str | Path, episode: str | int, camera: str) -> Path:
    dataset_name = Path(dataset_path).resolve().name or "xperience_10m"
    episode_name = str(episode)
    if not episode_name.startswith("ep"):
        episode_name = f"ep{episode_name}"
    return REPO_ROOT / "results" / "xperience_10m" / dataset_name / episode_name / camera


def save_xperience_predictions_npz(pred_hands: Dict, gt: Dict, output_path: str | Path) -> str:
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
    )
    return str(output_path)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate WiLoR on an xperience-10m episode.")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--episode", required=True, type=str)
    parser.add_argument("--camera", default="stereo_left", choices=sorted(CAMERA_FILES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--confidence_thresh", type=float, default=0.0)
    parser.add_argument("--det_conf", type=float, default=0.3)
    parser.add_argument("--img_focal", type=float, default=None, help="Override xperience-10m fx for WiLoR full-camera conversion.")
    parser.add_argument("--intrinsic_key", type=str, default="calibration/cam01/K")
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    paths = resolve_episode_paths(args.dataset_path, args.episode, args.camera)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataset_path, args.episode, args.camera)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_xperience_gt(
        paths.annotation_path,
        confidence_thresh=args.confidence_thresh,
        intrinsic_key=args.intrinsic_key,
    )
    video = egodex_utils.video_info(paths.video_path)
    hands = egodex_utils.selected_hands(args.hand)
    img_focal = float(args.img_focal) if args.img_focal is not None else float(gt["intrinsic"][0, 0])
    source_frame_indices = selected_source_frame_indices(video["frame_count"], args.max_frames, args.frame_stride)
    gt_visibility = gt_visibility_summary(gt, hands, frame_indices=source_frame_indices)

    pred_video_path = None if args.skip_pred_video else output_dir / "wilor_mano_overlay.mp4"
    pred_hands, run_info = egodex_utils.run_wilor_on_video(args, paths.video_path, pred_video_path, focal_length=img_focal)
    metrics = egodex_utils.evaluate_predictions(pred_hands, gt, hands, fps=float(video["fps"]) / args.frame_stride)

    outputs = {
        "wilor_mano_video": str(pred_video_path) if pred_video_path is not None else None,
        "predictions_npz": save_xperience_predictions_npz(pred_hands, gt, output_dir / "wilor_xperience_predictions.npz"),
    }
    if args.skip_gt_video:
        outputs["gt_skeleton_video"] = None
    else:
        gt_video_path = output_dir / "xperience_gt_skeleton.mp4"
        outputs["gt_skeleton_video"] = egodex_utils.render_gt_skeleton_video(
            paths.video_path,
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
            "hands": egodex_utils.save_joint_curve_outputs(pred_hands, gt, hands, plot_dir),
        }

    result = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "episode": paths.episode_dir.name,
        "camera": args.camera,
        "video_path": str(paths.video_path),
        "annotation_path": str(paths.annotation_path),
        "output_dir": str(output_dir),
        "video": video,
        "img_focal": img_focal,
        "frame_stride": args.frame_stride,
        "processed_source_frame_count": int(source_frame_indices.shape[0]),
        "processed_frames": run_info["frames"],
        "runtime": run_info,
        "gt_attrs": gt["attrs"],
        "gt_visibility": gt_visibility,
        "metrics": metrics,
        "outputs": outputs,
        "plots": plot_outputs,
    }
    output_json = Path(args.output_json) if args.output_json else output_dir / "xperience_wilor_eval.json"
    egodex_utils.write_json(output_json, result)

    print(f"xperience-10m {paths.episode_dir.name} WiLoR metrics:")
    print("  GT visibility:")
    for hand, visibility in gt_visibility.items():
        ranges = visibility["valid_ranges"][:5]
        ranges_text = ", ".join(f"{item['start']}-{item['end']}" for item in ranges) if ranges else "none"
        if len(visibility["valid_ranges"]) > len(ranges):
            ranges_text += ", ..."
        print(
            f"    {hand}: {visibility['valid_frames']} valid frame(s), "
            f"first={visibility['first_valid_frame']}, last={visibility['last_valid_frame']}, "
            f"ranges={ranges_text}"
        )
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
        print(f"Saved xperience-10m GT skeleton video to {outputs['gt_skeleton_video']['path']}")
    if plot_outputs["enabled"]:
        print(f"Saved joint comparison plots to {plot_outputs['dir']}")


if __name__ == "__main__":
    main()
