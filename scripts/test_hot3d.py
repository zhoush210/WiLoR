from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
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

from scripts import test_egodex as egodex_utils  # noqa: E402


DEFAULT_STREAM_ID = "214-1"
DEFAULT_FPS = 30.0
JOINT_NAMES = [f"joint_{idx:02d}" for idx in range(21)]
HOT3D_HAND_COLORS_BGR = {
    "left": (209, 153, 205),
    "right": (202, 152, 53),
}


def _read_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _frame_sort_key(frame_id: str) -> Tuple[int, int | str]:
    try:
        return (0, int(frame_id))
    except ValueError:
        return (1, frame_id)


def discover_hot3d_frame_ids(
    dataset_path: str | Path,
    stream_id: str = DEFAULT_STREAM_ID,
    max_frames: Optional[int] = None,
) -> List[str]:
    dataset_path = Path(dataset_path)
    image_paths = sorted(
        dataset_path.glob(f"*.image_{stream_id}.jpg"),
        key=lambda path: _frame_sort_key(path.name.split(".", 1)[0]),
    )
    if not image_paths:
        raise FileNotFoundError(f"No HOT3D images for stream '{stream_id}' under {dataset_path}")

    frame_ids = [path.name.split(".", 1)[0] for path in image_paths]
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("--max_frames must be a positive integer.")
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def _frame_image_path(dataset_path: str | Path, frame_id: str, stream_id: str) -> Path:
    return Path(dataset_path) / f"{frame_id}.image_{stream_id}.jpg"


def _camera_json_path(dataset_path: str | Path, frame_id: str) -> Path:
    return Path(dataset_path) / f"{frame_id}.cameras.json"


def _hands_json_path(dataset_path: str | Path, frame_id: str) -> Path:
    return Path(dataset_path) / f"{frame_id}.hands.json"


def _info_json_path(dataset_path: str | Path, frame_id: str) -> Path:
    return Path(dataset_path) / f"{frame_id}.info.json"


def intrinsic_from_hot3d_calibration(calibration: Optional[Dict]) -> np.ndarray:
    calibration = calibration or {}
    projection_params = calibration.get("projection_params") or []
    if len(projection_params) >= 3:
        focal = float(projection_params[0])
        cx = float(projection_params[1])
        cy = float(projection_params[2])
        return np.array(
            [
                [focal, 0.0, cx],
                [0.0, focal, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    fx = calibration.get("fx", calibration.get("focal_x"))
    fy = calibration.get("fy", calibration.get("focal_y", fx))
    if fx is None:
        width = calibration.get("image_width")
        height = calibration.get("image_height")
        if width is None or height is None:
            raise ValueError("HOT3D calibration is missing projection_params and image size.")
        fx = fy = max(float(width), float(height))
    width = float(calibration.get("image_width", 0.0))
    height = float(calibration.get("image_height", 0.0))
    cx = calibration.get("cx", width / 2.0)
    cy = calibration.get("cy", height / 2.0)
    return np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def project_hot3d_fisheye624(points_cam: np.ndarray, calibration: Dict) -> np.ndarray:
    calibration = calibration or {}
    projection_params = np.asarray(calibration.get("projection_params", []), dtype=np.float64).reshape(-1)
    if projection_params.shape[0] < 15:
        raise ValueError("HOT3D FISHEYE624 projection needs 15 projection_params values.")

    focal, cx, cy = projection_params[:3]
    radial = projection_params[3:9]
    p0, p1 = projection_params[9:11]
    s0, s1, s2, s3 = projection_params[11:15]

    points = np.asarray(points_cam, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_cam must have shape (N, 3).")

    projected = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & np.isfinite(z) & (z > 1e-8)
    if not np.any(valid):
        return projected.astype(np.float32)

    xy = points[valid, :2] / z[valid, None]
    x = xy[:, 0]
    y = xy[:, 1]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    theta2 = theta * theta
    theta_power = theta2.copy()
    radial_scale = np.ones_like(theta)
    for coeff in radial:
        radial_scale += coeff * theta_power
        theta_power *= theta2
    theta_distorted = theta * radial_scale

    scale = np.ones_like(r)
    nonzero = r > 1e-12
    scale[nonzero] = theta_distorted[nonzero] / r[nonzero]
    mx = x * scale
    my = y * scale

    rho2 = mx * mx + my * my
    rho4 = rho2 * rho2
    dx = 2.0 * p0 * mx * my + p1 * (rho2 + 2.0 * mx * mx) + s0 * rho2 + s1 * rho4
    dy = p0 * (rho2 + 2.0 * my * my) + 2.0 * p1 * mx * my + s2 * rho2 + s3 * rho4

    projected[valid] = np.stack(
        [
            focal * (mx + dx) + cx,
            focal * (my + dy) + cy,
        ],
        axis=1,
    )
    return projected.astype(np.float32)


def quaternion_wxyz_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def hot3d_pose_to_transform(pose: Dict) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = quaternion_wxyz_to_matrix(pose["quaternion_wxyz"])
    transform[:3, 3] = np.asarray(pose["translation_xyz"], dtype=np.float32).reshape(3)
    return transform


def load_hot3d_camera_data(
    dataset_path: str | Path,
    frame_ids: Sequence[str],
    stream_id: str = DEFAULT_STREAM_ID,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    transforms = []
    calibrations = []
    intrinsic = None
    for frame_id in frame_ids:
        camera_path = _camera_json_path(dataset_path, frame_id)
        if not camera_path.exists():
            raise FileNotFoundError(f"Missing HOT3D camera JSON: {camera_path}")
        cameras = _read_json(camera_path)
        if stream_id not in cameras:
            raise KeyError(f"Missing HOT3D stream '{stream_id}' in {camera_path}")

        camera = cameras[stream_id]
        transforms.append(hot3d_pose_to_transform(camera["T_world_from_camera"]))
        calibration = camera.get("calibration", {})
        calibrations.append(calibration)
        if intrinsic is None:
            intrinsic = intrinsic_from_hot3d_calibration(calibration)

    if intrinsic is None:
        raise ValueError("Need at least one HOT3D camera calibration.")
    return np.stack(transforms, axis=0).astype(np.float32), intrinsic, calibrations


def _empty_mano_param_arrays(frame_count: int) -> Dict:
    return {
        "root_orient": np.zeros((frame_count, 3), dtype=np.float32),
        "trans": np.zeros((frame_count, 3), dtype=np.float32),
        "pose_pca": np.zeros((frame_count, 15), dtype=np.float32),
        "betas": np.zeros((frame_count, 10), dtype=np.float32),
        "confidence": np.zeros(frame_count, dtype=np.float32),
        "valid": np.zeros(frame_count, dtype=bool),
    }


def load_hot3d_mano_params(
    dataset_path: str | Path,
    frame_ids: Sequence[str],
    stream_id: str = DEFAULT_STREAM_ID,
    confidence_thresh: float = 0.0,
) -> Dict[str, Dict]:
    frame_count = len(frame_ids)
    params = {
        "left": _empty_mano_param_arrays(frame_count),
        "right": _empty_mano_param_arrays(frame_count),
    }

    for frame_idx, frame_id in enumerate(frame_ids):
        hands_path = _hands_json_path(dataset_path, frame_id)
        if not hands_path.exists():
            continue
        hands_json = _read_json(hands_path)

        for hand in ("left", "right"):
            hand_json = hands_json.get(hand)
            if not hand_json or "mano_pose" not in hand_json:
                continue

            mano_pose = hand_json["mano_pose"]
            pose_pca = np.asarray(mano_pose.get("thetas", []), dtype=np.float32).reshape(-1)
            wrist_xform = np.asarray(mano_pose.get("wrist_xform", []), dtype=np.float32).reshape(-1)
            if pose_pca.shape[0] != 15:
                raise ValueError(
                    f"Expected 15 HOT3D MANO PCA values for {hand} frame {frame_id}, got {pose_pca.shape[0]}."
                )
            if wrist_xform.shape[0] != 6:
                raise ValueError(
                    f"Expected 6 HOT3D wrist_xform values for {hand} frame {frame_id}, got {wrist_xform.shape[0]}."
                )

            betas_finite = True
            betas = hand_json.get("betas", mano_pose.get("betas"))
            if betas is not None:
                betas = np.asarray(betas, dtype=np.float32).reshape(-1)
                if betas.shape[0] < 10:
                    raise ValueError(
                        f"Expected at least 10 HOT3D MANO betas for {hand} frame {frame_id}, got {betas.shape[0]}."
                    )
                params[hand]["betas"][frame_idx] = betas[:10]
                betas_finite = bool(np.isfinite(betas[:10]).all())

            visibility = float(hand_json.get("visibilities_modeled", {}).get(stream_id, 1.0))
            target = params[hand]
            target["root_orient"][frame_idx] = wrist_xform[:3]
            target["trans"][frame_idx] = wrist_xform[3:6]
            target["pose_pca"][frame_idx] = pose_pca
            target["confidence"][frame_idx] = visibility
            target["valid"][frame_idx] = (
                np.isfinite(wrist_xform).all()
                and np.isfinite(pose_pca).all()
                and betas_finite
                and visibility >= float(confidence_thresh)
            )
    return params


def _mano_model_candidates(hand: str, mano_model_folder: Optional[str | Path] = None) -> List[Path]:
    filename = "MANO_LEFT.pkl" if hand == "left" else "MANO_RIGHT.pkl"
    if mano_model_folder is not None:
        model_path = Path(mano_model_folder).expanduser().resolve()
        return [model_path if model_path.suffix == ".pkl" else model_path / filename]

    if hand == "left":
        return [
            REPO_ROOT / "mano_data" / filename,
            Path("/home/unitree/HaWoR/_DATA/data_left/mano_left") / filename,
            Path("/home/unitree/HaWoR/_DATA/data/mano") / filename,
        ]
    return [
        REPO_ROOT / "mano_data" / filename,
        Path("/home/unitree/HaWoR/_DATA/data/mano") / filename,
    ]


def resolve_mano_model_file(hand: str, mano_model_folder: Optional[str | Path] = None) -> Path:
    candidates = _mano_model_candidates(hand, mano_model_folder)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {hand} MANO model. Checked: "
        + ", ".join(str(path) for path in candidates)
        + ". Pass --mano_model_folder if your MANO files live elsewhere."
    )


def _load_hot3d_pca_mano(hand: str, mano_model_folder: Optional[str | Path], use_cuda: bool):
    import torch
    import smplx

    model_path = resolve_mano_model_file(hand, mano_model_folder)
    mano = smplx.create(
        str(model_path),
        "mano",
        use_pca=True,
        is_rhand=(hand == "right"),
        num_pca_comps=15,
        create_body_pose=False,
    )
    if use_cuda and torch.cuda.is_available():
        mano = mano.cuda()
    mano.eval()
    return mano


def _load_hot3d_rotmat_mano_layer(hand: str, mano_model_folder: Optional[str | Path], use_cuda: bool):
    import torch
    from wilor.models.mano_wrapper import MANO

    model_path = resolve_mano_model_file(hand, mano_model_folder)
    mano = MANO(
        model_path=str(model_path),
        use_pca=False,
        is_rhand=(hand == "right"),
        create_body_pose=False,
    )
    if hand == "left" and hasattr(mano, "shapedirs"):
        mano.shapedirs[:, 0, :] *= -1
    if use_cuda and torch.cuda.is_available():
        mano = mano.cuda()
    mano.eval()
    return mano


def hot3d_mano_params_to_world_joints(
    params: Dict[str, Dict],
    mano_model_folder: Optional[str | Path] = None,
    use_cuda: bool = True,
) -> Dict[str, Dict]:
    import torch
    from wilor.utils.geometry import aa_to_rotmat

    use_cuda = bool(use_cuda and torch.cuda.is_available())
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    hands = {}

    for hand in ("left", "right"):
        hand_params = params[hand]
        valid = np.asarray(hand_params["valid"], dtype=bool)
        frame_count = int(valid.shape[0])
        joints = np.full((frame_count, 21, 3), np.nan, dtype=np.float32)
        vertices = np.full((frame_count, 778, 3), np.nan, dtype=np.float32)
        faces = None

        if np.any(valid):
            pca_mano = _load_hot3d_pca_mano(hand, mano_model_folder, use_cuda=use_cuda)
            mano = _load_hot3d_rotmat_mano_layer(hand, mano_model_folder, use_cuda=use_cuda)
            faces = np.asarray(mano.faces, dtype=np.int32)
            valid_indices = np.flatnonzero(valid)
            root_orient = torch.from_numpy(hand_params["root_orient"][valid]).float().to(device)
            pose_pca = torch.from_numpy(hand_params["pose_pca"][valid]).float().to(device)
            betas = torch.from_numpy(hand_params["betas"][valid]).float().to(device)
            trans = torch.from_numpy(hand_params["trans"][valid]).float().to(device)
            with torch.no_grad():
                pca_output = pca_mano(
                    global_orient=root_orient,
                    hand_pose=pose_pca,
                    betas=betas,
                    transl=trans,
                    return_verts=False,
                    return_full_pose=True,
                )
                full_pose = pca_output.full_pose
                root_rot = aa_to_rotmat(full_pose[:, :3]).view(-1, 1, 3, 3)
                hand_rot = aa_to_rotmat(full_pose[:, 3:].reshape(-1, 3)).view(-1, 15, 3, 3)
                output = mano(
                    global_orient=root_rot,
                    hand_pose=hand_rot,
                    betas=betas,
                    transl=trans,
                    return_verts=True,
                )
            joints[valid_indices] = output.joints.detach().cpu().numpy().astype(np.float32)
            vertices[valid_indices] = output.vertices.detach().cpu().numpy().astype(np.float32)

        hands[hand] = {
            "joints_world": joints,
            "vertices_world": vertices,
            "faces": faces,
            "confidence": hand_params["confidence"].astype(np.float32),
            "valid": valid,
            "joint_names": egodex_utils.egodex_joint_names(hand),
        }
    return hands


def hot3d_world_to_camera_joints(joints_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    return egodex_utils.world_to_camera_joints(joints_world, camera_transforms)


def _load_hot3d_attrs(dataset_path: str | Path, frame_ids: Sequence[str], stream_id: str) -> Dict:
    attrs = {
        "clip_id": Path(dataset_path).name,
        "stream_id": stream_id,
        "frame_count": len(frame_ids),
        "frame_ids": list(frame_ids),
    }
    if frame_ids:
        info_path = _info_json_path(dataset_path, frame_ids[0])
        if info_path.exists():
            attrs.update(_read_json(info_path))
    return attrs


def load_hot3d_gt(
    dataset_path: str | Path,
    stream_id: str = DEFAULT_STREAM_ID,
    confidence_thresh: float = 0.0,
    max_frames: Optional[int] = None,
    frame_ids: Optional[Sequence[str]] = None,
    mano_model_folder: Optional[str | Path] = None,
    use_cuda: bool = True,
) -> Dict:
    if frame_ids is None:
        frame_ids = discover_hot3d_frame_ids(dataset_path, stream_id, max_frames=max_frames)
    frame_ids = list(frame_ids)
    camera_transforms, intrinsic, camera_calibrations = load_hot3d_camera_data(dataset_path, frame_ids, stream_id)
    mano_params = load_hot3d_mano_params(
        dataset_path,
        frame_ids,
        stream_id=stream_id,
        confidence_thresh=confidence_thresh,
    )
    hands_world = hot3d_mano_params_to_world_joints(
        mano_params,
        mano_model_folder=mano_model_folder,
        use_cuda=use_cuda,
    )

    hands = {}
    for hand, hand_data in hands_world.items():
        hands[hand] = dict(hand_data)
        hands[hand]["joints"] = hot3d_world_to_camera_joints(hand_data["joints_world"], camera_transforms)
        hands[hand]["vertices"] = hot3d_world_to_camera_joints(hand_data["vertices_world"], camera_transforms)

    return {
        "intrinsic": intrinsic,
        "camera_calibrations": camera_calibrations,
        "camera_transforms": camera_transforms,
        "attrs": _load_hot3d_attrs(dataset_path, frame_ids, stream_id),
        "hands": hands,
    }


def default_output_dir(dataset_path: str | Path, stream_id: str = DEFAULT_STREAM_ID) -> Path:
    clip_name = Path(dataset_path).resolve().name or "hot3d"
    safe_stream = str(stream_id).replace("/", "_")
    return REPO_ROOT / "results" / "hot3d" / clip_name / safe_stream


def _copy_or_link(src: str | Path, dst: str | Path) -> None:
    src = Path(src).resolve()
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_video_from_images(video_path: str | Path, image_dir: str | Path, frame_count: int, fps: float) -> Dict:
    image_paths = [Path(image_dir) / f"{idx:04d}.jpg" for idx in range(frame_count)]
    if not image_paths:
        raise ValueError("Need at least one HOT3D frame to create a workspace video.")

    first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Could not read HOT3D image: {image_paths[0]}")
    height, width = first_image.shape[:2]
    writer = egodex_utils.open_video_writer(video_path, fps, width, height)
    try:
        writer.write(first_image)
        for image_path in image_paths[1:]:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read HOT3D image: {image_path}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    finally:
        writer.release()

    return {
        "frame_count": int(frame_count),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
    }


def _existing_workspace_ready(video_path: str | Path, image_dir: str | Path, frame_count: int) -> bool:
    if not Path(video_path).exists():
        return False
    image_files = sorted(Path(image_dir).glob("*.jpg"))
    return len(image_files) >= int(frame_count)


def _workspace_video_stem(dataset_path: str | Path, stream_id: str, frame_count: int, total_frames: int) -> str:
    safe_stream = str(stream_id).replace("/", "_")
    stem = f"{Path(dataset_path).name}_{safe_stream}"
    if int(frame_count) != int(total_frames):
        stem = f"{stem}_{frame_count}f"
    return stem


def prepare_hot3d_workspace(
    dataset_path: str | Path,
    stream_id: str = DEFAULT_STREAM_ID,
    output_dir: Optional[str | Path] = None,
    max_source_frames: Optional[int] = None,
    fps: float = DEFAULT_FPS,
) -> Tuple[Path, Path, Dict, List[str]]:
    dataset_path = Path(dataset_path)
    all_frame_ids = discover_hot3d_frame_ids(dataset_path, stream_id)
    frame_ids = all_frame_ids
    if max_source_frames is not None:
        max_source_frames = int(max_source_frames)
        if max_source_frames <= 0:
            raise ValueError("--max_frames must be a positive integer.")
        frame_ids = all_frame_ids[:max_source_frames]
    if not frame_ids:
        raise ValueError(f"No frames selected for HOT3D clip: {dataset_path}")

    root = Path(output_dir) if output_dir else default_output_dir(dataset_path, stream_id)
    workspace_root = root / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    stem = _workspace_video_stem(dataset_path, stream_id, len(frame_ids), len(all_frame_ids))
    video_path = workspace_root / f"{stem}.mp4"
    seq_folder = video_path.with_suffix("")
    image_dir = seq_folder / "extracted_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if not _existing_workspace_ready(video_path, image_dir, len(frame_ids)):
        for old_image in image_dir.glob("*.jpg"):
            old_image.unlink()
        for out_idx, frame_id in enumerate(frame_ids):
            source_image = _frame_image_path(dataset_path, frame_id, stream_id)
            if not source_image.exists():
                raise FileNotFoundError(f"Missing HOT3D image: {source_image}")
            _copy_or_link(source_image, image_dir / f"{out_idx:04d}.jpg")
        video_info = _write_video_from_images(video_path, image_dir, len(frame_ids), fps=fps)
    else:
        video_info = egodex_utils.video_info(video_path)

    return video_path, seq_folder, video_info, list(frame_ids)


def selected_source_frame_indices(frame_count: int, max_frames: Optional[int], frame_stride: int) -> np.ndarray:
    if frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    indices = np.arange(0, int(frame_count), int(frame_stride), dtype=np.int32)
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    return indices


def compact_frame_ranges(frame_indices: np.ndarray) -> List[Dict[str, int]]:
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


def _hand_color_bgr(hand: str) -> Tuple[int, int, int]:
    return HOT3D_HAND_COLORS_BGR.get(hand, egodex_utils.HAND_COLORS_BGR[hand])


def draw_fisheye_skeleton(
    frame_bgr: np.ndarray,
    joints_camera: np.ndarray,
    calibration: Dict,
    color_bgr: Tuple[int, int, int],
    label: Optional[str] = None,
) -> None:
    uv = project_hot3d_fisheye624(joints_camera, calibration)
    valid = np.isfinite(joints_camera).all(axis=1) & (np.asarray(joints_camera)[:, 2] > 1e-6)
    height, width = frame_bgr.shape[:2]
    in_frame = (
        valid
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    points = np.round(np.nan_to_num(uv, nan=-1e6)).astype(np.int32)
    for start, end in egodex_utils.HAND_EDGES:
        if in_frame[start] and in_frame[end]:
            cv2.line(frame_bgr, tuple(points[start]), tuple(points[end]), color_bgr, 2, cv2.LINE_AA)
    for joint_i, point in enumerate(points):
        if in_frame[joint_i]:
            radius = 4 if joint_i == 0 else 3
            cv2.circle(frame_bgr, tuple(point), radius, color_bgr, -1, cv2.LINE_AA)
    if label:
        valid_points = points[in_frame]
        if valid_points.size:
            cv2.putText(frame_bgr, label, tuple(valid_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)


def _draw_fisheye_mano_camera(
    image: np.ndarray,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    calibration: Dict,
    color_bgr: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    height, width = image.shape[:2]
    vertices_camera = np.asarray(vertices_camera, dtype=np.float64)
    projected = project_hot3d_fisheye624(vertices_camera, calibration)

    faces = np.asarray(faces, dtype=np.int32)
    z = vertices_camera[:, 2]
    finite = np.isfinite(projected).all(axis=1) & np.isfinite(z)
    face_valid = finite[faces].all(axis=1) & (z[faces] > 1e-5).all(axis=1)
    if not np.any(face_valid):
        return image

    valid_faces = faces[face_valid]
    depth_order = np.argsort(z[valid_faces].mean(axis=1))[::-1]
    overlay = image.copy()
    margin = float(max(width, height))
    min_xy = np.array([-margin, -margin], dtype=np.float32)
    max_xy = np.array([width + margin, height + margin], dtype=np.float32)

    for face in valid_faces[depth_order]:
        points = projected[face]
        if (
            points[:, 0].max() < -margin
            or points[:, 0].min() > width + margin
            or points[:, 1].max() < -margin
            or points[:, 1].min() > height + margin
        ):
            continue
        points = np.clip(points, min_xy, max_xy)
        cv2.fillConvexPoly(
            overlay,
            np.rint(points).astype(np.int32),
            color=tuple(int(v) for v in color_bgr),
            lineType=cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0)


def run_wilor_on_hot3d_video(
    args,
    video_path: str | Path,
    output_video_path: Optional[str | Path],
    camera_calibrations: Sequence[Dict],
    focal_length: Optional[float] = None,
) -> Tuple[Dict, Dict]:
    runtime = egodex_utils._load_wilor_runtime(args)
    torch = runtime["torch"]
    DataLoader = runtime["DataLoader"]
    ViTDetDataset = runtime["ViTDetDataset"]
    recursive_to = runtime["recursive_to"]
    cam_crop_to_full = runtime["cam_crop_to_full"]
    model = runtime["model"]
    model_cfg = runtime["model_cfg"]
    detector = runtime["detector"]
    device = runtime["device"]

    info = egodex_utils.video_info(video_path)
    full_focal_length = float(focal_length) if focal_length is not None else (
        model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(info["width"], info["height"])
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = None
    if output_video_path is not None:
        writer = egodex_utils.open_video_writer(output_video_path, info["fps"] / args.frame_stride, info["width"], info["height"])

    predictions = egodex_utils._empty_prediction_store()
    faces = np.asarray(model.mano.faces, dtype=np.int32)
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

            egodex_utils._append_empty_frame(predictions, frame_idx)
            detections = detector(frame, conf=args.det_conf, verbose=False)[0]
            boxes, right, scores = egodex_utils._detections_to_arrays(detections, args.det_conf)
            render_items = []

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
                        vertices_camera = verts + cam_t[None, :]
                        joints_camera = joints + cam_t[None, :]
                        egodex_utils._set_frame_prediction(
                            predictions,
                            hand,
                            joints_camera,
                            boxes[person_id],
                            float(scores[person_id]),
                        )
                        render_items.append(
                            {
                                "hand": hand,
                                "vertices_camera": vertices_camera,
                                "joints_camera": joints_camera,
                            }
                        )

            if writer is not None:
                output_frame = frame.copy()
                calibration = camera_calibrations[min(frame_idx, len(camera_calibrations) - 1)]
                for item in render_items:
                    hand = item["hand"]
                    output_frame = _draw_fisheye_mano_camera(
                        output_frame,
                        item["vertices_camera"],
                        faces,
                        calibration,
                        _hand_color_bgr(hand),
                    )
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

    return egodex_utils._finalize_predictions(predictions), {
        "frames": processed,
        "rendered_frames": written,
        "device": str(device),
        "focal_length": full_focal_length,
        "projection": "HOT3D FISHEYE624 projection_params",
    }


def render_hot3d_gt_mano_video(
    video_path: str | Path,
    gt: Dict,
    hands: Sequence[str],
    output_path: str | Path,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> Dict:
    info = egodex_utils.video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    writer = egodex_utils.open_video_writer(output_path, info["fps"] / frame_stride, info["width"], info["height"])
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

            calibration = gt["camera_calibrations"][min(frame_idx, len(gt["camera_calibrations"]) - 1)]
            output_frame = frame.copy()
            for hand in hands:
                hand_data = gt["hands"][hand]
                faces = hand_data.get("faces")
                vertices = hand_data.get("vertices")
                valid = hand_data.get("valid")
                if faces is None or vertices is None or valid is None:
                    continue
                if frame_idx >= vertices.shape[0] or frame_idx >= valid.shape[0] or not bool(valid[frame_idx]):
                    continue
                output_frame = _draw_fisheye_mano_camera(
                    output_frame,
                    vertices[frame_idx],
                    faces,
                    calibration,
                    _hand_color_bgr(hand),
                )
            writer.write(output_frame)
            frame_idx += 1
            written += 1
    finally:
        cap.release()
        writer.release()
    return {"path": str(output_path), "frames": written, "fps": info["fps"] / frame_stride}


def _projection_params_array(camera_calibrations: Sequence[Dict]) -> np.ndarray:
    values = []
    for calibration in camera_calibrations:
        params = np.asarray(calibration.get("projection_params", []), dtype=np.float32).reshape(-1)
        if params.size:
            values.append(params)
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    length = min(value.shape[0] for value in values)
    return np.stack([value[:length] for value in values], axis=0).astype(np.float32)


def save_hot3d_predictions_npz(
    pred_hands: Dict,
    gt: Dict,
    frame_ids: Sequence[str],
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frame_ids=np.asarray(frame_ids),
        pred_left_joints=pred_hands["left"]["joints"],
        pred_left_valid=pred_hands["left"]["valid"],
        pred_left_frame_indices=pred_hands["left"]["frame_indices"],
        pred_right_joints=pred_hands["right"]["joints"],
        pred_right_valid=pred_hands["right"]["valid"],
        pred_right_frame_indices=pred_hands["right"]["frame_indices"],
        gt_left_joints=gt["hands"]["left"]["joints"],
        gt_left_joints_world=gt["hands"]["left"]["joints_world"],
        gt_left_vertices=gt["hands"]["left"]["vertices"],
        gt_left_vertices_world=gt["hands"]["left"]["vertices_world"],
        gt_left_valid=gt["hands"]["left"]["valid"],
        gt_right_joints=gt["hands"]["right"]["joints"],
        gt_right_joints_world=gt["hands"]["right"]["joints_world"],
        gt_right_vertices=gt["hands"]["right"]["vertices"],
        gt_right_vertices_world=gt["hands"]["right"]["vertices_world"],
        gt_right_valid=gt["hands"]["right"]["valid"],
        camera_intrinsic=gt["intrinsic"],
        camera_transforms=gt["camera_transforms"],
        projection_params=_projection_params_array(gt["camera_calibrations"]),
    )
    return str(output_path)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate WiLoR on a HOT3D clip.")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--stream_id", default=DEFAULT_STREAM_ID, type=str, help="HOT3D image stream to use, e.g. 214-1, 1201-1, 1201-2.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument("--mano_model_folder", type=str, default=None, help="Optional MANO folder containing MANO_LEFT.pkl and MANO_RIGHT.pkl for HOT3D GT conversion.")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--confidence_thresh", type=float, default=0.0)
    parser.add_argument("--det_conf", type=float, default=0.3)
    parser.add_argument("--img_focal", type=float, default=None, help="Override HOT3D projection_params[0] for WiLoR full-camera conversion.")
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="FPS for the generated HOT3D workspace video.")
    parser.add_argument("--fast", action="store_true", help="Use FP16 and WiLoR backbone block skipping.")
    parser.add_argument("--compile_backbone", action="store_true", help="Also torch.compile the WiLoR backbone when --fast is set.")
    parser.add_argument("--cpu", action="store_true", help="Run WiLoR on CPU even when CUDA is available.")
    parser.add_argument("--cpu_mano", action="store_true", help="Run HOT3D GT MANO conversion on CPU.")
    parser.add_argument("--skip_pred_video", action="store_true")
    parser.add_argument("--skip_gt_video", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args(argv)


def _summary_mean_text(summary: Dict) -> str:
    mean = summary["mean"]
    return "nan" if mean is None else f"{mean:.6f}"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(dataset_path, args.stream_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_source_frames = args.max_frames * args.frame_stride if args.max_frames is not None else None
    workspace_video_path, seq_folder, video_info, frame_ids = prepare_hot3d_workspace(
        dataset_path,
        stream_id=args.stream_id,
        output_dir=output_dir,
        max_source_frames=max_source_frames,
        fps=args.fps,
    )
    gt = load_hot3d_gt(
        dataset_path,
        stream_id=args.stream_id,
        confidence_thresh=args.confidence_thresh,
        frame_ids=frame_ids,
        mano_model_folder=args.mano_model_folder,
        use_cuda=not args.cpu_mano,
    )

    hands = egodex_utils.selected_hands(args.hand)
    img_focal = float(args.img_focal) if args.img_focal is not None else float(gt["intrinsic"][0, 0])
    source_frame_indices = selected_source_frame_indices(video_info["frame_count"], args.max_frames, args.frame_stride)
    gt_visibility = gt_visibility_summary(gt, hands, frame_indices=source_frame_indices)

    pred_video_path = None if args.skip_pred_video else output_dir / "wilor_hot3d_fisheye_overlay.mp4"
    pred_hands, run_info = run_wilor_on_hot3d_video(
        args,
        workspace_video_path,
        pred_video_path,
        gt["camera_calibrations"],
        focal_length=img_focal,
    )
    metrics = egodex_utils.evaluate_predictions(pred_hands, gt, hands, fps=float(video_info["fps"]) / args.frame_stride)

    outputs = {
        "wilor_fisheye_video": str(pred_video_path) if pred_video_path is not None else None,
        "predictions_npz": save_hot3d_predictions_npz(
            pred_hands,
            gt,
            frame_ids,
            output_dir / "wilor_hot3d_predictions.npz",
        ),
    }
    if args.skip_gt_video:
        outputs["gt_mano_video"] = None
    else:
        gt_video_path = output_dir / "hot3d_gt_fisheye_mano.mp4"
        outputs["gt_mano_video"] = render_hot3d_gt_mano_video(
            workspace_video_path,
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

    first_calibration = gt["camera_calibrations"][0] if gt["camera_calibrations"] else {}
    result = {
        "dataset_path": str(dataset_path.resolve()),
        "clip_id": dataset_path.name,
        "stream_id": args.stream_id,
        "video_path": str(workspace_video_path),
        "workspace_video_path": str(workspace_video_path),
        "sequence_folder": str(seq_folder),
        "output_dir": str(output_dir),
        "video": video_info,
        "img_focal": img_focal,
        "projection_model_type": first_calibration.get("projection_model_type"),
        "projection_params": first_calibration.get("projection_params"),
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
    output_json = Path(args.output_json) if args.output_json else output_dir / "hot3d_wilor_eval.json"
    egodex_utils.write_json(output_json, result)

    print(f"HOT3D clip {dataset_path.name} stream {args.stream_id} WiLoR metrics:")
    print("  GT visibility:")
    for hand, visibility in gt_visibility.items():
        ranges = visibility["valid_ranges"][:5]
        ranges_text = ", ".join(f"{item['start']}-{item['end']}" for item in ranges) if ranges else "none"
        if len(visibility["valid_ranges"]) > len(ranges):
            ranges_text += ", ..."
        print(
            f"    {hand}: {visibility['valid_frames']} valid frame(s), "
            f"first={visibility['first_valid_frame']}, last={visibility['last_valid_frame']}, ranges={ranges_text}"
        )
    for name, summary in metrics["overall"].items():
        print(f"  overall {name}: {_summary_mean_text(summary)}")
    for hand, hand_result in metrics["hands"].items():
        print(f"  {hand}: {hand_result['valid_frames']} valid frame(s)")
        for name, summary in hand_result["metrics"].items():
            print(f"    {name}: {_summary_mean_text(summary)}")
    print(f"Saved results to {output_json}")
    if pred_video_path is not None:
        print(f"Saved WiLoR fisheye overlay video to {pred_video_path}")
    if outputs["gt_mano_video"]:
        print(f"Saved HOT3D GT fisheye MANO video to {outputs['gt_mano_video']['path']}")
    if plot_outputs["enabled"]:
        print(f"Saved joint comparison plots to {plot_outputs['dir']}")


if __name__ == "__main__":
    main()
