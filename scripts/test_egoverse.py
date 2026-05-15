from __future__ import annotations

import argparse
from copy import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Dict, Optional, Sequence

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


MISSING_SHARD_ENTRY = 2**64 - 1
DEFAULT_IMAGE_KEY = "images.front_1"


@dataclass(frozen=True)
class PreparedVideo:
    video_path: Path
    frame_indices: np.ndarray
    fps: float
    width: int
    height: int


def read_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def root_attrs(dataset_path: str | Path) -> Dict:
    metadata_path = Path(dataset_path) / "zarr.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing egoVerse zarr group metadata: {metadata_path}")
    return read_json(metadata_path).get("attributes", {})


def array_metadata(array_path: str | Path) -> Dict:
    metadata_path = Path(array_path) / "zarr.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing egoVerse zarr array metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        raise ValueError(f"Unsupported zarr array metadata in {metadata_path}")
    return metadata


def dtype_from_zarr(data_type: str) -> np.dtype:
    if data_type == "float64":
        return np.dtype("<f8")
    if data_type == "float32":
        return np.dtype("<f4")
    if data_type in {"int64", "uint64", "int32", "uint32", "uint8"}:
        return np.dtype("<" + np.dtype(data_type).str[1:])
    raise ValueError(f"Unsupported egoVerse zarr data_type: {data_type}")


def sharding_config(metadata: Dict) -> Dict:
    codecs = metadata.get("codecs") or []
    if len(codecs) != 1 or codecs[0].get("name") != "sharding_indexed":
        raise ValueError("Only zarr v3 sharding_indexed arrays are supported for egoVerse.")
    return codecs[0]["configuration"]


def chunk_key(array_path: str | Path, outer_coords: Sequence[int], separator: str = "/") -> Path:
    coord_parts = [str(int(coord)) for coord in outer_coords]
    if separator == "/":
        return Path(array_path, "c", *coord_parts)
    return Path(array_path, "c", separator.join(coord_parts))


def inner_chunk_counts(outer_chunk_shape: Sequence[int], inner_chunk_shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        int(math.ceil(float(outer) / float(inner)))
        for outer, inner in zip(outer_chunk_shape, inner_chunk_shape)
    )


def index_trailer_size(sharding_conf: Dict, entry_count: int) -> int:
    index_size = int(entry_count) * 16
    for codec in sharding_conf.get("index_codecs", []):
        if codec.get("name") == "crc32c":
            index_size += 4
    return index_size


def decode_payload(encoded: bytes, codecs: Sequence[Dict]) -> bytes:
    payload = encoded
    for codec in reversed(codecs):
        name = codec.get("name")
        if name == "zstd":
            try:
                import zstandard as zstd
            except ImportError as exc:
                raise ImportError(
                    "scripts/test_egoverse.py requires zstandard to decode egoVerse zarr chunks. "
                    "Install it in the active wilor environment with: python -m pip install zstandard"
                ) from exc
            payload = zstd.ZstdDecompressor().decompress(payload)
        elif name in {"bytes", "vlen-bytes"}:
            continue
        else:
            raise ValueError(f"Unsupported egoVerse zarr codec: {name}")
    return payload


def decode_vlen_bytes(raw: bytes) -> list[bytes]:
    if len(raw) < 4:
        return []
    count = struct.unpack_from("<I", raw, 0)[0]
    offset = 4
    lengths = []
    for _ in range(count):
        if offset + 4 > len(raw):
            raise ValueError("Malformed vlen-bytes chunk in egoVerse zarr array.")
        lengths.append(struct.unpack_from("<I", raw, offset)[0])
        offset += 4

    values = []
    for length in lengths:
        end = offset + int(length)
        if end > len(raw):
            raise ValueError("Malformed vlen-bytes payload in egoVerse zarr array.")
        values.append(raw[offset:end])
        offset = end
    return values


def entry_coordinates(entry_idx: int, inner_counts: Sequence[int]) -> tuple[int, ...]:
    coords = []
    remaining = int(entry_idx)
    for count in reversed(inner_counts):
        coords.append(remaining % count)
        remaining //= count
    return tuple(reversed(coords))


def iter_outer_coords(shape: Sequence[int], outer_chunk_shape: Sequence[int]):
    counts = [
        int(math.ceil(float(dim) / float(chunk)))
        for dim, chunk in zip(shape, outer_chunk_shape)
    ]
    return np.ndindex(*counts)


def fill_numeric_chunk(
    output: np.ndarray,
    chunk_values: np.ndarray,
    start: Sequence[int],
    inner_chunk_shape: Sequence[int],
) -> None:
    slices = []
    source_slices = []
    for dim_idx, start_idx in enumerate(start):
        dim_end = min(start_idx + inner_chunk_shape[dim_idx], output.shape[dim_idx])
        if dim_end <= start_idx:
            return
        slices.append(slice(start_idx, dim_end))
        source_slices.append(slice(0, dim_end - start_idx))
    output[tuple(slices)] = chunk_values[tuple(source_slices)]


def reshape_numeric_chunk(raw: bytes, dtype: np.dtype, inner_chunk_shape: Sequence[int]) -> np.ndarray:
    values = np.frombuffer(raw, dtype=dtype)
    expected = int(math.prod(inner_chunk_shape))
    if values.size == expected:
        return values.reshape(inner_chunk_shape)
    if len(inner_chunk_shape) == 1:
        return values.reshape(-1)

    trailing = int(math.prod(inner_chunk_shape[1:]))
    if trailing <= 0 or values.size % trailing != 0:
        raise ValueError(
            f"Cannot reshape egoVerse zarr chunk with {values.size} values "
            f"to inner shape {tuple(inner_chunk_shape)}."
        )
    return values.reshape((values.size // trailing, *inner_chunk_shape[1:]))


def load_sharded_zarr_array(array_path: str | Path, limit: Optional[int] = None):
    array_path = Path(array_path)
    metadata = array_metadata(array_path)
    shape = tuple(int(value) for value in metadata["shape"])
    if limit is not None and shape:
        shape = (min(shape[0], int(limit)), *shape[1:])

    sharding_conf = sharding_config(metadata)
    outer_chunk_shape = tuple(
        int(value)
        for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    inner_chunk_shape = tuple(int(value) for value in sharding_conf["chunk_shape"])
    separator = (
        metadata.get("chunk_key_encoding", {})
        .get("configuration", {})
        .get("separator", "/")
    )
    inner_counts = inner_chunk_counts(outer_chunk_shape, inner_chunk_shape)
    entry_count = int(math.prod(inner_counts))
    index_size = index_trailer_size(sharding_conf, entry_count)
    codecs = sharding_conf.get("codecs", [])

    if metadata["data_type"] == "variable_length_bytes":
        values = [b""] * (shape[0] if shape else 0)
        is_vlen = True
        dtype = None
    else:
        dtype = dtype_from_zarr(metadata["data_type"])
        values = np.full(shape, metadata.get("fill_value", 0), dtype=dtype)
        is_vlen = False

    for outer_coords in iter_outer_coords(shape, outer_chunk_shape):
        shard_path = chunk_key(array_path, outer_coords, separator=separator)
        if not shard_path.exists():
            continue
        shard = shard_path.read_bytes()
        if len(shard) < index_size:
            raise ValueError(f"Malformed sharded zarr file: {shard_path}")
        index = shard[-index_size:]
        index_codecs = sharding_conf.get("index_codecs", [])
        if index_codecs and index_codecs[-1].get("name") == "crc32c":
            index = index[:-4]

        for entry_idx in range(entry_count):
            offset, length = struct.unpack_from("<QQ", index, entry_idx * 16)
            if offset == MISSING_SHARD_ENTRY and length == MISSING_SHARD_ENTRY:
                continue

            inner_coords = entry_coordinates(entry_idx, inner_counts)
            start = tuple(
                outer_coords[dim] * outer_chunk_shape[dim]
                + inner_coords[dim] * inner_chunk_shape[dim]
                for dim in range(len(shape))
            )
            if any(start[dim] >= shape[dim] for dim in range(len(shape))):
                continue

            raw = decode_payload(shard[offset : offset + length], codecs)
            if is_vlen:
                chunk_values = decode_vlen_bytes(raw)
                for value_idx, value in enumerate(chunk_values):
                    row_idx = start[0] + value_idx
                    if row_idx < len(values):
                        values[row_idx] = value
            else:
                chunk_values = reshape_numeric_chunk(raw, dtype, inner_chunk_shape)
                fill_numeric_chunk(values, chunk_values, start, inner_chunk_shape)

    return values


def normalize_keypoints(keypoints: np.ndarray, name: str) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim == 2:
        if keypoints.shape[1] == 63:
            return keypoints.reshape(-1, 21, 3)
        if keypoints.shape == (21, 3):
            return keypoints[None, ...]
        if keypoints.shape == (3, 21):
            return keypoints.T[None, ...]
    if keypoints.ndim == 3:
        if keypoints.shape[-2:] == (21, 3):
            return keypoints
        if keypoints.shape[-2:] == (3, 21):
            return np.swapaxes(keypoints, -1, -2)
        if keypoints.shape[-1] == 63:
            return keypoints.reshape(keypoints.shape[0], 21, 3)
    raise ValueError(f"Unsupported {name} shape {keypoints.shape}")


def keypoint_valid_mask(joints: np.ndarray, confidence_thresh: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joints, dtype=np.float32)
    finite = np.isfinite(joints).all(axis=(1, 2))
    nonzero = np.linalg.norm(joints.reshape(joints.shape[0], -1), axis=1) > 1e-9
    confidence = np.ones(joints.shape[0], dtype=np.float32)
    return confidence, finite & nonzero & (confidence >= float(confidence_thresh))


def first_present(mapping: Dict, *keys: str):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def intrinsic_matrix_from_mapping(value: Dict, width: Optional[int] = None, height: Optional[int] = None):
    if not isinstance(value, dict):
        return None

    fx = first_present(value, "fx", "fl_x", "focal_x", "focal_length_x")
    fy = first_present(value, "fy", "fl_y", "focal_y", "focal_length_y")
    if fx is None:
        return None
    if fy is None:
        fy = fx

    src_width = first_present(value, "w", "width", "image_width")
    src_height = first_present(value, "h", "height", "image_height")
    scale_x = 1.0
    scale_y = 1.0
    if width is not None and src_width not in (None, 0):
        scale_x = float(width) / float(src_width)
    if height is not None and src_height not in (None, 0):
        scale_y = float(height) / float(src_height)

    cx = first_present(value, "cx", "principal_x", "principal_point_x")
    cy = first_present(value, "cy", "principal_y", "principal_point_y")
    cx = float(cx) * scale_x if cx is not None else (float(width) / 2.0 if width else 0.0)
    cy = float(cy) * scale_y if cy is not None else (float(height) / 2.0 if height else 0.0)

    return np.array(
        [
            [float(fx) * scale_x, 0.0, cx],
            [0.0, float(fy) * scale_y, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def intrinsic_from_attrs(attrs: Dict, width: Optional[int] = None, height: Optional[int] = None):
    intrinsics = attrs.get("intrinsics")
    if not isinstance(intrinsics, dict) or not intrinsics:
        return None

    intrinsic = intrinsic_matrix_from_mapping(intrinsics, width=width, height=height)
    if intrinsic is not None:
        return intrinsic

    for value in intrinsics.values():
        intrinsic = intrinsic_matrix_from_mapping(value, width=width, height=height)
        if intrinsic is not None:
            return intrinsic
    return None


def fallback_intrinsic(width: Optional[int], height: Optional[int], focal: Optional[float] = None) -> np.ndarray:
    if width is None or height is None:
        raise ValueError("Cannot build fallback intrinsic without decoded egoVerse image dimensions.")
    focal = float(focal) if focal is not None else max(600.0, float(width))
    return np.array(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"Expected quaternion array with shape (T, 4), got {quaternion.shape}.")

    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-8
    safe = np.zeros_like(quaternion, dtype=np.float32)
    safe[:, 0] = 1.0
    safe[valid] = quaternion[valid] / norm[valid]

    w = safe[:, 0]
    x = safe[:, 1]
    y = safe[:, 2]
    z = safe[:, 3]

    matrix = np.empty((safe.shape[0], 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - z * w)
    matrix[:, 0, 2] = 2.0 * (x * z + y * w)
    matrix[:, 1, 0] = 2.0 * (x * y + z * w)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - x * w)
    matrix[:, 2, 0] = 2.0 * (x * z - y * w)
    matrix[:, 2, 1] = 2.0 * (y * z + x * w)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def pose7_to_camera_transforms(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected pose array with shape (T, 7), got {pose.shape}.")

    transforms = np.tile(np.eye(4, dtype=np.float32), (pose.shape[0], 1, 1))
    transforms[:, :3, :3] = quaternion_wxyz_to_matrix(pose[:, 3:7])
    transforms[:, :3, 3] = pose[:, :3]
    return transforms


def identity_camera_transforms(frame_count: int) -> np.ndarray:
    return np.tile(np.eye(4, dtype=np.float32), (int(frame_count), 1, 1))


def load_head_camera_transforms(
    dataset_path: str | Path,
    frame_count: int,
    limit: Optional[int] = None,
    frame_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    head_pose_path = Path(dataset_path) / "obs_head_pose"
    if not (head_pose_path / "zarr.json").exists():
        return identity_camera_transforms(frame_count)

    head_pose = load_sharded_zarr_array(head_pose_path, limit=limit)
    head_pose = np.asarray(head_pose, dtype=np.float32)
    if head_pose.size == 0:
        return identity_camera_transforms(frame_count)
    transforms = pose7_to_camera_transforms(head_pose)
    if frame_indices is not None:
        return transforms[frame_indices[:frame_count]]
    return transforms[:frame_count]


def egoverse_world_to_camera_joints(joints_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    return egodex_utils.world_to_camera_joints(joints_world, camera_transforms)


def image_dimensions_from_metadata(dataset_path: str | Path, attrs: Dict, image_key: str = DEFAULT_IMAGE_KEY):
    width = height = None
    feature_shape = attrs.get("features", {}).get(image_key, {}).get("shape")
    if feature_shape and len(feature_shape) >= 2:
        height, width = feature_shape[:2]
        return int(width), int(height)

    image_meta_path = Path(dataset_path) / image_key / "zarr.json"
    if image_meta_path.exists():
        image_shape = read_json(image_meta_path).get("shape", [])
        if len(image_shape) >= 3:
            height, width = image_shape[1:3]
    if width is None or height is None:
        return None, None
    return int(width), int(height)


def select_frame_indices(total_frames: int, max_frames: Optional[int] = None, frame_stride: int = 1) -> np.ndarray:
    if frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    indices = np.arange(0, int(total_frames), int(frame_stride), dtype=np.int32)
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("--max_frames must be a positive integer.")
        indices = indices[:max_frames]
    if indices.size == 0:
        raise ValueError("No egoVerse frames selected.")
    return indices


def decode_jpeg(frame_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode egoVerse JPEG frame.")
    return image


def write_frames_to_video(video_path: str | Path, frame_bytes: Sequence[bytes], fps: float) -> Dict:
    if not frame_bytes:
        raise ValueError("Need at least one egoVerse frame to create a video.")

    first_image = decode_jpeg(frame_bytes[0])
    height, width = first_image.shape[:2]
    writer = egodex_utils.open_video_writer(video_path, fps=float(fps), width=width, height=height)
    written = 0
    try:
        writer.write(first_image)
        written += 1
        for frame in frame_bytes[1:]:
            image = decode_jpeg(frame)
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
            written += 1
    finally:
        writer.release()
    return {"frame_count": written, "fps": float(fps), "width": width, "height": height}


def default_output_dir(dataset_path: str | Path) -> Path:
    episode_id = Path(dataset_path).resolve().name or "egoverse"
    return REPO_ROOT / "results" / "egoverse" / episode_id


def prepare_egoverse_video(
    dataset_path: str | Path,
    output_dir: str | Path,
    image_key: str = DEFAULT_IMAGE_KEY,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> PreparedVideo:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    image_array_path = dataset_path / image_key
    image_meta = array_metadata(image_array_path)
    total_frames = int(image_meta["shape"][0])
    frame_indices = select_frame_indices(total_frames, max_frames=max_frames, frame_stride=frame_stride)
    load_limit = int(frame_indices[-1]) + 1
    frame_bytes_all = load_sharded_zarr_array(image_array_path, limit=load_limit)
    selected_bytes = [frame_bytes_all[int(frame_idx)] for frame_idx in frame_indices]

    attrs = root_attrs(dataset_path)
    source_fps = float(attrs.get("fps") or 30.0)
    output_fps = source_fps / float(frame_stride)
    stem = image_key.replace(".", "_").replace("/", "_")
    video_path = output_dir / f"{stem}.mp4"
    info = write_frames_to_video(video_path, selected_bytes, fps=output_fps)
    return PreparedVideo(
        video_path=video_path,
        frame_indices=frame_indices,
        fps=float(info["fps"]),
        width=int(info["width"]),
        height=int(info["height"]),
    )


def load_egoverse_gt(
    dataset_path: str | Path,
    confidence_thresh: float = 0.0,
    image_key: str = DEFAULT_IMAGE_KEY,
    frame_indices: Optional[np.ndarray] = None,
    image_size: Optional[tuple[int, int]] = None,
) -> Dict:
    dataset_path = Path(dataset_path)
    attrs = root_attrs(dataset_path)
    limit = int(frame_indices[-1]) + 1 if frame_indices is not None and len(frame_indices) else None
    left_joints = normalize_keypoints(
        load_sharded_zarr_array(dataset_path / "left.obs_keypoints", limit=limit),
        "left.obs_keypoints",
    )
    right_joints = normalize_keypoints(
        load_sharded_zarr_array(dataset_path / "right.obs_keypoints", limit=limit),
        "right.obs_keypoints",
    )
    frame_count = min(left_joints.shape[0], right_joints.shape[0])
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices, dtype=np.int32)
        frame_indices = frame_indices[frame_indices < frame_count]
        left_joints = left_joints[frame_indices]
        right_joints = right_joints[frame_indices]
        frame_count = len(frame_indices)
    else:
        left_joints = left_joints[:frame_count]
        right_joints = right_joints[:frame_count]

    left_conf, left_valid = keypoint_valid_mask(left_joints, confidence_thresh)
    right_conf, right_valid = keypoint_valid_mask(right_joints, confidence_thresh)

    width, height = image_dimensions_from_metadata(dataset_path, attrs, image_key=image_key)
    if (width is None or height is None) and image_size is not None:
        width, height = image_size
    intrinsic = intrinsic_from_attrs(attrs, width=width, height=height)
    if intrinsic is None and image_size is not None:
        intrinsic = fallback_intrinsic(width=image_size[0], height=image_size[1])

    camera_transforms = load_head_camera_transforms(
        dataset_path,
        frame_count,
        limit=limit,
        frame_indices=frame_indices,
    )
    left_joints_world = left_joints.astype(np.float32)
    right_joints_world = right_joints.astype(np.float32)
    left_joints_camera = egoverse_world_to_camera_joints(left_joints_world, camera_transforms)
    right_joints_camera = egoverse_world_to_camera_joints(right_joints_world, camera_transforms)
    return {
        "intrinsic": intrinsic,
        "camera_transforms": camera_transforms,
        "attrs": attrs,
        "hands": {
            "left": {
                "joints": left_joints_camera,
                "joints_world": left_joints_world,
                "confidence": left_conf,
                "valid": left_valid,
                "joint_names": egodex_utils.egodex_joint_names("left"),
            },
            "right": {
                "joints": right_joints_camera,
                "joints_world": right_joints_world,
                "confidence": right_conf,
                "valid": right_valid,
                "joint_names": egodex_utils.egodex_joint_names("right"),
            },
        },
    }


def resolve_img_focal(args, gt: Dict, video: PreparedVideo) -> float:
    if args.img_focal is not None:
        return float(args.img_focal)
    if gt.get("intrinsic") is not None:
        return float(gt["intrinsic"][0, 0])
    fallback = max(600.0, float(video.width))
    print(f"No egoVerse intrinsic focal length found; using {fallback:.1f}")
    return fallback


def save_egoverse_predictions_npz(
    pred_hands: Dict,
    gt: Dict,
    source_frame_indices: np.ndarray,
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_frame_indices=np.asarray(source_frame_indices, dtype=np.int32),
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


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate WiLoR on an egoVerse episode.")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--image_key", default=DEFAULT_IMAGE_KEY, type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--confidence_thresh", type=float, default=0.0)
    parser.add_argument("--det_conf", type=float, default=0.3)
    parser.add_argument("--img_focal", type=float, default=None, help="Override egoVerse fx for WiLoR full-camera conversion.")
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
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_video = prepare_egoverse_video(
        args.dataset_path,
        output_dir=output_dir,
        image_key=args.image_key,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    gt = load_egoverse_gt(
        args.dataset_path,
        confidence_thresh=args.confidence_thresh,
        image_key=args.image_key,
        frame_indices=prepared_video.frame_indices,
        image_size=(prepared_video.width, prepared_video.height),
    )
    if gt["intrinsic"] is None:
        gt["intrinsic"] = fallback_intrinsic(prepared_video.width, prepared_video.height, args.img_focal)

    hands = egodex_utils.selected_hands(args.hand)
    img_focal = resolve_img_focal(args, gt, prepared_video)

    runner_args = copy(args)
    runner_args.frame_stride = 1
    runner_args.max_frames = None

    pred_video_path = None if args.skip_pred_video else output_dir / "wilor_mano_overlay.mp4"
    pred_hands, run_info = egodex_utils.run_wilor_on_video(
        runner_args,
        prepared_video.video_path,
        pred_video_path,
        focal_length=img_focal,
    )
    metrics = egodex_utils.evaluate_predictions(pred_hands, gt, hands, fps=prepared_video.fps)

    outputs = {
        "input_video": str(prepared_video.video_path),
        "wilor_mano_video": str(pred_video_path) if pred_video_path is not None else None,
        "predictions_npz": save_egoverse_predictions_npz(
            pred_hands,
            gt,
            prepared_video.frame_indices,
            output_dir / "wilor_egoverse_predictions.npz",
        ),
    }
    if args.skip_gt_video:
        outputs["gt_skeleton_video"] = None
    else:
        gt_video_path = output_dir / "egoverse_gt_skeleton.mp4"
        outputs["gt_skeleton_video"] = egodex_utils.render_gt_skeleton_video(
            prepared_video.video_path,
            gt,
            hands,
            gt_video_path,
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

    attrs = gt["attrs"]
    result = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "episode_id": str(attrs.get("episode_id") or Path(args.dataset_path).name),
        "image_key": args.image_key,
        "output_dir": str(output_dir),
        "video": {
            "frame_count": int(len(prepared_video.frame_indices)),
            "fps": float(prepared_video.fps),
            "width": int(prepared_video.width),
            "height": int(prepared_video.height),
        },
        "source_frame_indices": prepared_video.frame_indices.astype(int).tolist(),
        "img_focal": img_focal,
        "frame_stride": args.frame_stride,
        "processed_frames": run_info["frames"],
        "runtime": run_info,
        "gt_attrs": attrs,
        "metrics": metrics,
        "outputs": outputs,
        "plots": plot_outputs,
    }
    output_json = Path(args.output_json) if args.output_json else output_dir / "egoverse_wilor_eval.json"
    egodex_utils.write_json(output_json, result)

    print(f"egoVerse {result['episode_id']} WiLoR metrics:")
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
    print(f"Saved egoVerse input video to {prepared_video.video_path}")
    if pred_video_path is not None:
        print(f"Saved WiLoR MANO overlay video to {pred_video_path}")
    if outputs["gt_skeleton_video"]:
        print(f"Saved egoVerse GT skeleton video to {outputs['gt_skeleton_video']['path']}")
    if plot_outputs["enabled"]:
        print(f"Saved joint comparison plots to {plot_outputs['dir']}")


if __name__ == "__main__":
    main()
