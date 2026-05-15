import unittest
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_egodex import (
    build_joint_curve_data,
    build_h264_transcode_command,
    compute_accel_errors_m_per_s2,
    compute_mpjpe_errors,
    evaluate_predictions,
    gt_skeleton_label,
    world_to_camera_joints,
)


class EgoDexMetricTests(unittest.TestCase):
    def test_mpjpe_errors_use_valid_frames_without_alignment(self):
        pred = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]],
                [[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]],
            ],
            dtype=np.float32,
        )
        gt = np.zeros_like(pred)
        valid = np.array([True, False, True])

        errors = compute_mpjpe_errors(pred, gt, valid)

        np.testing.assert_allclose(errors, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_accel_errors_are_second_difference_scaled_by_fps_squared(self):
        gt = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(4, 1, 1)
        gt = np.pad(gt, ((0, 0), (0, 0), (0, 2)))
        pred = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32).reshape(4, 1, 1)
        pred = np.pad(pred, ((0, 0), (0, 0), (0, 2)))

        errors = compute_accel_errors_m_per_s2(pred, gt, np.ones(4, dtype=bool), fps=2.0)

        np.testing.assert_allclose(errors, np.array([4.0, 4.0]))

    def test_accel_errors_require_valid_three_frame_windows(self):
        pred = np.zeros((4, 1, 3), dtype=np.float32)
        gt = np.zeros_like(pred)
        valid = np.array([True, False, True, True])

        errors = compute_accel_errors_m_per_s2(pred, gt, valid, fps=30.0)

        self.assertEqual(errors.size, 0)

    def test_evaluate_predictions_reports_mpjpe_in_millimeters_and_accel_in_meters(self):
        pred_joints = np.array(
            [
                [[0.000, 0.0, 0.0]],
                [[0.001, 0.0, 0.0]],
                [[0.003, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        gt_joints = np.zeros_like(pred_joints)
        pred_hands = {
            "right": {
                "joints": pred_joints,
                "valid": np.ones(3, dtype=bool),
                "frame_indices": np.arange(3, dtype=np.int32),
            }
        }
        gt = {
            "hands": {
                "right": {
                    "joints": gt_joints,
                    "valid": np.ones(3, dtype=bool),
                }
            }
        }

        metrics = evaluate_predictions(pred_hands, gt, ["right"], fps=2.0)

        self.assertIn("mpjpe_mm", metrics["overall"])
        self.assertIn("accel_m_per_s2", metrics["overall"])
        self.assertNotIn("mpjpe", metrics["overall"])
        self.assertNotIn("accel_mm_per_s2", metrics["overall"])
        self.assertAlmostEqual(metrics["overall"]["mpjpe_mm"]["mean"], 4.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["overall"]["accel_m_per_s2"]["mean"], 0.004, places=6)

    def test_joint_curve_errors_are_in_millimeters(self):
        pred_hands = {
            "left": {
                "joints": np.array([[[0.002, 0.0, 0.0]]], dtype=np.float32),
                "valid": np.array([True]),
                "frame_indices": np.array([0], dtype=np.int32),
            }
        }
        gt = {
            "hands": {
                "left": {
                    "joints": np.zeros((1, 1, 3), dtype=np.float32),
                    "valid": np.array([True]),
                    "joint_names": ["wrist"],
                }
            }
        }

        curve_data = build_joint_curve_data(pred_hands, gt, "left")

        np.testing.assert_allclose(curve_data["error_mm"], np.array([[2.0]], dtype=np.float32))

    def test_gt_skeleton_labels_are_disabled_by_default(self):
        self.assertIsNone(gt_skeleton_label("left"))
        self.assertEqual(gt_skeleton_label("right", show_labels=True), "GT right")

    def test_world_to_camera_joints_uses_camera_to_world_transform_inverse(self):
        joints_world = np.array([[[2.0, 3.0, 4.0]]], dtype=np.float32)
        camera_transforms = np.eye(4, dtype=np.float32)[None]
        camera_transforms[0, :3, 3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        joints_camera = world_to_camera_joints(joints_world, camera_transforms)

        np.testing.assert_allclose(joints_camera, np.array([[[1.0, 2.0, 3.0]]]))

    def test_h264_transcode_command_targets_vscode_compatible_mp4(self):
        command = build_h264_transcode_command("raw.mp4", "final.mp4")

        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertEqual(command[-1], "final.mp4")


if __name__ == "__main__":
    unittest.main()
