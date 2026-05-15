import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "test_hot3d.py"


def load_module():
    spec = importlib.util.spec_from_file_location("test_hot3d", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Hot3DScriptTests(unittest.TestCase):
    def test_discover_frame_ids_sorts_numeric_ids_and_applies_limit(self):
        mod = load_module()
        with self.subTest("numeric sort"):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                for frame_id in ("10", "2", "001"):
                    (root / f"{frame_id}.image_214-1.jpg").write_bytes(b"")

                self.assertEqual(
                    mod.discover_hot3d_frame_ids(root, stream_id="214-1", max_frames=2),
                    ["001", "2"],
                )

    def test_intrinsic_from_projection_params_uses_hot3d_fisheye_center(self):
        mod = load_module()
        calibration = {
            "projection_params": [609.0, 707.5, 702.25, 0.1],
            "image_width": 1408,
            "image_height": 1408,
        }

        intrinsic = mod.intrinsic_from_hot3d_calibration(calibration)

        np.testing.assert_allclose(
            intrinsic,
            np.array(
                [
                    [609.0, 0.0, 707.5],
                    [0.0, 609.0, 702.25],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_project_fisheye624_maps_optical_axis_to_principal_point(self):
        mod = load_module()
        calibration = {"projection_params": [600.0, 700.0, 710.0] + [0.0] * 12}

        projected = mod.project_hot3d_fisheye624(np.array([[0.0, 0.0, 2.0]], dtype=np.float32), calibration)

        np.testing.assert_allclose(projected, np.array([[700.0, 710.0]], dtype=np.float32))

    def test_world_joints_are_converted_to_selected_camera_frame(self):
        mod = load_module()
        joints_world = np.array([[[2.0, 3.0, 4.0]]], dtype=np.float32)
        camera_transforms = np.eye(4, dtype=np.float32)[None]
        camera_transforms[0, :3, 3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        joints_camera = mod.hot3d_world_to_camera_joints(joints_world, camera_transforms)

        np.testing.assert_allclose(joints_camera, np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32))

    def test_gt_video_renderer_uses_mano_mesh_not_skeleton(self):
        import cv2

        mod = load_module()
        with TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "input.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (10, 10),
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
            writer.release()

            calls = []

            def fake_draw_mano(image, vertices_camera, faces, calibration, color_bgr):
                calls.append((vertices_camera.copy(), faces.copy(), dict(calibration), tuple(color_bgr)))
                return image

            def fail_skeleton(*args, **kwargs):
                raise AssertionError("GT video should render MANO mesh, not skeleton joints")

            class FakeWriter:
                def __init__(self):
                    self.frames = 0

                def write(self, frame):
                    self.frames += 1

                def release(self):
                    pass

            original_draw_mano = mod._draw_fisheye_mano_camera
            original_draw_skeleton = mod.draw_fisheye_skeleton
            original_open_writer = mod.egodex_utils.open_video_writer
            fake_writer = FakeWriter()
            try:
                mod._draw_fisheye_mano_camera = fake_draw_mano
                mod.draw_fisheye_skeleton = fail_skeleton
                mod.egodex_utils.open_video_writer = lambda *args, **kwargs: fake_writer
                result = mod.render_hot3d_gt_mano_video(
                    video_path,
                    {
                        "camera_calibrations": [{"projection_params": [100.0, 5.0, 5.0] + [0.0] * 12}],
                        "hands": {
                            "right": {
                                "vertices": np.array(
                                    [[
                                        [0.0, 0.0, 1.0],
                                        [0.1, 0.0, 1.0],
                                        [0.0, 0.1, 1.0],
                                    ]],
                                    dtype=np.float32,
                                ),
                                "faces": np.array([[0, 1, 2]], dtype=np.int32),
                                "valid": np.array([True]),
                            }
                        },
                    },
                    ["right"],
                    Path(tmp) / "gt.mp4",
                )
            finally:
                mod._draw_fisheye_mano_camera = original_draw_mano
                mod.draw_fisheye_skeleton = original_draw_skeleton
                mod.egodex_utils.open_video_writer = original_open_writer

            self.assertEqual(result["frames"], 1)
            self.assertEqual(fake_writer.frames, 1)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(calls[0][1], np.array([[0, 1, 2]], dtype=np.int32))

    def test_prediction_video_renderer_uses_mano_mesh_not_skeleton(self):
        import cv2
        import torch

        mod = load_module()
        with TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "input.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (10, 10),
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
            writer.release()

            batch = {
                "right": torch.tensor([1.0], dtype=torch.float32),
                "box_center": torch.tensor([[5.0, 5.0]], dtype=torch.float32),
                "box_size": torch.tensor([4.0], dtype=torch.float32),
                "img_size": torch.tensor([[10.0, 10.0]], dtype=torch.float32),
                "img": torch.zeros((1, 3, 256, 256), dtype=torch.float32),
                "personid": torch.tensor([0], dtype=torch.int64),
            }

            class FakeBoxes:
                data = torch.tensor([[1.0, 1.0, 4.0, 4.0, 0.9, 1.0]], dtype=torch.float32)

            class FakeDetector:
                def __call__(self, frame, conf, verbose):
                    return [SimpleNamespace(boxes=FakeBoxes())]

            class FakeDataLoader:
                def __init__(self, *args, **kwargs):
                    pass

                def __iter__(self):
                    return iter([batch])

            class FakeModel:
                mano = SimpleNamespace(faces=np.array([[0, 1, 2]], dtype=np.int32))

                def __call__(self, batch_arg):
                    return {
                        "pred_cam": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                        "pred_vertices": torch.zeros((1, 778, 3), dtype=torch.float32),
                        "pred_keypoints_3d": torch.zeros((1, 21, 3), dtype=torch.float32),
                    }

            class FakeWriter:
                def __init__(self):
                    self.frames = 0

                def write(self, frame):
                    self.frames += 1

                def release(self):
                    pass

            calls = []

            def fake_draw_mano(image, vertices_camera, faces, calibration, color_bgr):
                calls.append((vertices_camera.copy(), faces.copy(), dict(calibration), tuple(color_bgr)))
                return image

            def fail_skeleton(*args, **kwargs):
                raise AssertionError("Prediction overlay should render MANO mesh, not skeleton joints")

            runtime = {
                "torch": torch,
                "DataLoader": FakeDataLoader,
                "ViTDetDataset": lambda *args, **kwargs: object(),
                "recursive_to": lambda value, device: value,
                "cam_crop_to_full": lambda *args, **kwargs: torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
                "model": FakeModel(),
                "model_cfg": SimpleNamespace(
                    EXTRA=SimpleNamespace(FOCAL_LENGTH=5000.0),
                    MODEL=SimpleNamespace(IMAGE_SIZE=256.0),
                ),
                "detector": FakeDetector(),
                "device": "cpu",
            }

            original_runtime = mod.egodex_utils._load_wilor_runtime
            original_open_writer = mod.egodex_utils.open_video_writer
            original_draw_mano = mod._draw_fisheye_mano_camera
            original_draw_skeleton = mod.draw_fisheye_skeleton
            fake_writer = FakeWriter()
            try:
                mod.egodex_utils._load_wilor_runtime = lambda args: runtime
                mod.egodex_utils.open_video_writer = lambda *args, **kwargs: fake_writer
                mod._draw_fisheye_mano_camera = fake_draw_mano
                mod.draw_fisheye_skeleton = fail_skeleton
                _, run_info = mod.run_wilor_on_hot3d_video(
                    SimpleNamespace(
                        frame_stride=1,
                        max_frames=1,
                        det_conf=0.1,
                        rescale_factor=2.0,
                        batch_size=1,
                        fast=False,
                    ),
                    video_path,
                    Path(tmp) / "pred.mp4",
                    [{"projection_params": [100.0, 5.0, 5.0] + [0.0] * 12}],
                    focal_length=100.0,
                )
            finally:
                mod.egodex_utils._load_wilor_runtime = original_runtime
                mod.egodex_utils.open_video_writer = original_open_writer
                mod._draw_fisheye_mano_camera = original_draw_mano
                mod.draw_fisheye_skeleton = original_draw_skeleton

            self.assertEqual(run_info["rendered_frames"], 1)
            self.assertEqual(fake_writer.frames, 1)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(calls[0][1], np.array([[0, 1, 2]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
