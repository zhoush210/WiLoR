import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "test_egoverse.py"


def load_module():
    spec = importlib.util.spec_from_file_location("test_egoverse", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EgoverseScriptTests(unittest.TestCase):
    def test_default_output_dir_uses_episode_id(self):
        mod = load_module()

        output_dir = mod.default_output_dir("/data/egoVerse/69b57e518cd7957ebe4794cd")

        self.assertEqual(
            output_dir,
            REPO_ROOT / "results" / "egoverse" / "69b57e518cd7957ebe4794cd",
        )

    def test_intrinsic_mapping_scales_to_decoded_image_size(self):
        mod = load_module()

        intrinsic = mod.intrinsic_matrix_from_mapping(
            {
                "w": 1920,
                "h": 1080,
                "fl_x": 760.5,
                "fl_y": 759.0,
                "cx": 960.0,
                "cy": 540.0,
            },
            width=640,
            height=360,
        )

        np.testing.assert_allclose(
            intrinsic,
            np.array(
                [
                    [253.5, 0.0, 320.0],
                    [0.0, 253.0, 180.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_normalize_keypoints_accepts_flat_63_vectors(self):
        mod = load_module()
        keypoints = np.arange(2 * 63, dtype=np.float32).reshape(2, 63)

        normalized = mod.normalize_keypoints(keypoints, "left.obs_keypoints")

        self.assertEqual(normalized.shape, (2, 21, 3))
        np.testing.assert_array_equal(normalized[0, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(normalized[1, 20], np.array([123.0, 124.0, 125.0], dtype=np.float32))

    def test_world_joints_are_converted_with_camera_to_world_inverse(self):
        mod = load_module()
        joints_world = np.array([[[2.0, 3.0, 4.0]]], dtype=np.float32)
        camera_transforms = np.eye(4, dtype=np.float32)[None]
        camera_transforms[0, :3, 3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        joints_camera = mod.egoverse_world_to_camera_joints(joints_world, camera_transforms)

        np.testing.assert_allclose(joints_camera, np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
