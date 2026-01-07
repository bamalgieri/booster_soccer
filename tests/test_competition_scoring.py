import unittest

from training_scripts.competition_scoring import (
    aggregate_competition_reports,
    compute_competition_report,
)


class CompetitionScoringTests(unittest.TestCase):
    def test_report_penalty_kick_totals(self) -> None:
        reward_terms = {
            "robot_distance_ball": 2.0,
            "ball_vel_twd_goal": 1.0,
            "goal_scored": 1.0,
            "offside": 0.0,
            "ball_hits": 0.0,
            "robot_fallen": 0.0,
            "ball_blocked": 0.0,
        }
        report = compute_competition_report(
            reward_terms=reward_terms,
            task_name="goalie_penalty_kick",
            steps=10,
        )
        self.assertAlmostEqual(report["comp_steps"], -1.0)
        self.assertAlmostEqual(report["comp_total"], 3.5)
        self.assertEqual(report["missing_component_count"], 0)

    def test_missing_components_counted(self) -> None:
        reward_terms = {"goal_scored": 1.0}
        report = compute_competition_report(
            reward_terms=reward_terms,
            task_name="goalie_penalty_kick",
            steps=3,
        )
        self.assertAlmostEqual(report["comp_goal_scored"], 2.5)
        self.assertAlmostEqual(report["comp_steps"], -1.0)
        self.assertAlmostEqual(report["comp_total"], 1.5)
        self.assertEqual(report["missing_component_count"], 6)

    def test_bool_values_converted(self) -> None:
        reward_terms = {"success": True, "distance": 2.0, "offside": False}
        report = compute_competition_report(
            reward_terms=reward_terms,
            task_name="kick_to_target",
            steps=2,
        )
        self.assertAlmostEqual(report["comp_success"], 2.0)
        self.assertAlmostEqual(report["comp_offside"], 0.0)
        self.assertAlmostEqual(report["comp_distance"], 1.0)
        self.assertAlmostEqual(report["comp_steps"], -0.3)
        self.assertAlmostEqual(report["comp_total"], 2.7)

    def test_steps_constant_per_episode(self) -> None:
        report = compute_competition_report(
            reward_terms={},
            task_name="kick_to_target",
            steps=5000,
        )
        self.assertAlmostEqual(report["comp_steps"], -0.3)

    def test_unknown_task_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_competition_report({}, task_name="unknown", steps=1)

    def test_aggregate_reports_with_missing_components(self) -> None:
        reports = [
            compute_competition_report(
                reward_terms={"goal_scored": 1.0},
                task_name="goalie_penalty_kick",
                steps=2,
            ),
            compute_competition_report(
                reward_terms={},
                task_name="goalie_penalty_kick",
                steps=2,
            ),
        ]
        aggregated = aggregate_competition_reports("goalie_penalty_kick", reports)
        self.assertIn("comp_robot_distance_ball", aggregated)
        self.assertIn("comp_ball_vel_twd_goal", aggregated)
        self.assertIn("comp_goal_scored", aggregated)
        self.assertIn("comp_steps", aggregated)
        self.assertIn("comp_total", aggregated)
        self.assertIn("missing_component_count", aggregated)

    def test_aggregate_reports_empty_list(self) -> None:
        aggregated = aggregate_competition_reports("kick_to_target", [])
        self.assertIn("comp_total", aggregated)
        self.assertIn("comp_steps", aggregated)


if __name__ == "__main__":
    unittest.main()
