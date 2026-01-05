import tempfile
import unittest

import numpy as np

from booster_control.macro_playback import load_kick_macro, trigger_macro


class DummyLowerControl:
    def __init__(self, dof_count: int = 12) -> None:
        self.default_dof_pos = np.zeros(dof_count, dtype=np.float32)
        self.cfg = {
            "control": {"action_scale": 0.5},
            "normalization": {"clip_actions": 1.0},
        }


class MacroPlaybackTests(unittest.TestCase):
    def test_trigger_macro_requires_proximity(self) -> None:
        command = np.array([1.0, -1.0, 1.0], dtype=np.float32)
        info_far = {
            "ball_xpos_rel_robot": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "goal_team_0_rel_robot": np.array([2.0, 0.0, 0.0], dtype=np.float32),
        }
        self.assertFalse(trigger_macro(command, info_far, ball_radius=0.6))

        info_near = {
            "ball_xpos_rel_robot": np.array([0.1, 0.0, 0.0], dtype=np.float32),
            "goal_team_0_rel_robot": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
        self.assertTrue(trigger_macro(command, info_near, ball_radius=0.6))

    def test_load_kick_macro_respects_max_steps(self) -> None:
        dummy = DummyLowerControl()
        qpos = np.zeros((5, 7 + len(dummy.default_dof_pos)), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/macro.npz"
            np.savez(path, qpos=qpos)
            macro = load_kick_macro(path, dummy, max_steps=2)

        self.assertEqual(macro.actions.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
