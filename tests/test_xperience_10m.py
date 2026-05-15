import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "test_xperience_10m.py"


def load_module():
    spec = importlib.util.spec_from_file_location("test_xperience_10m", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Xperience10mScriptTests(unittest.TestCase):
    def test_resolve_episode_paths_uses_ep_folder_and_default_camera(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode_dir = root / "ep2"
            episode_dir.mkdir()
            (episode_dir / "annotation.hdf5").write_bytes(b"")
            (episode_dir / "stereo_left.mp4").write_bytes(b"")

            resolved = mod.resolve_episode_paths(root, 2, "stereo_left")

            self.assertEqual(resolved.episode_dir, episode_dir)
            self.assertEqual(resolved.video_path, episode_dir / "stereo_left.mp4")
            self.assertEqual(resolved.annotation_path, episode_dir / "annotation.hdf5")

    def test_intrinsic_vector_to_matrix(self):
        mod = load_module()

        intrinsic = mod.intrinsic_vector_to_matrix(np.array([200.0, 201.0, 256.0, 255.0]))

        np.testing.assert_allclose(
            intrinsic,
            np.array([[200.0, 0.0, 256.0], [0.0, 201.0, 255.0], [0.0, 0.0, 1.0]]),
        )

    def test_load_xperience_gt_filters_all_zero_hands(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "annotation.hdf5"
            with h5py.File(annotation_path, "w") as h5:
                calibration = h5.create_group("calibration")
                cam01 = calibration.create_group("cam01")
                cam01.create_dataset("K", data=np.array([200.0, 200.0, 256.0, 256.0]))
                hand_mocap = h5.create_group("hand_mocap")
                left = np.zeros((3, 21, 3), dtype=np.float32)
                right = np.ones((3, 21, 3), dtype=np.float32)
                right[1] = 0.0
                hand_mocap.create_dataset("left_joints_3d", data=left)
                hand_mocap.create_dataset("right_joints_3d", data=right)

            gt = mod.load_xperience_gt(annotation_path)

            np.testing.assert_allclose(gt["intrinsic"], np.array([[200.0, 0.0, 256.0], [0.0, 200.0, 256.0], [0.0, 0.0, 1.0]]))
            np.testing.assert_array_equal(gt["hands"]["left"]["valid"], np.array([False, False, False]))
            np.testing.assert_array_equal(gt["hands"]["right"]["valid"], np.array([True, False, True]))

    def test_valid_frame_ranges_compacts_boolean_mask(self):
        mod = load_module()

        ranges = mod.valid_frame_ranges(np.array([False, True, True, False, True, False, True, True, True]))

        self.assertEqual(
            ranges,
            [
                {"start": 1, "end": 2, "count": 2},
                {"start": 4, "end": 4, "count": 1},
                {"start": 6, "end": 8, "count": 3},
            ],
        )

    def test_gt_visibility_summary_reports_first_and_last_valid_frame(self):
        mod = load_module()
        gt = {
            "hands": {
                "left": {"valid": np.array([False, False, True, True, False])},
                "right": {"valid": np.array([True, True, False, False, False])},
            }
        }

        summary = mod.gt_visibility_summary(gt, ["left", "right"])

        self.assertEqual(summary["left"]["valid_frames"], 2)
        self.assertEqual(summary["left"]["first_valid_frame"], 2)
        self.assertEqual(summary["left"]["last_valid_frame"], 3)
        self.assertEqual(summary["right"]["valid_ranges"], [{"start": 0, "end": 1, "count": 2}])

    def test_gt_visibility_summary_uses_selected_source_frames(self):
        mod = load_module()
        gt = {
            "hands": {
                "left": {"valid": np.array([False, True, True, False, True])},
            }
        }

        summary = mod.gt_visibility_summary(gt, ["left"], frame_indices=np.array([0, 2, 4]))

        self.assertEqual(summary["left"]["valid_frames"], 2)
        self.assertEqual(summary["left"]["first_valid_frame"], 2)
        self.assertEqual(summary["left"]["last_valid_frame"], 4)
        self.assertEqual(
            summary["left"]["valid_ranges"],
            [
                {"start": 2, "end": 2, "count": 1},
                {"start": 4, "end": 4, "count": 1},
            ],
        )


if __name__ == "__main__":
    unittest.main()
