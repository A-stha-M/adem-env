"""
Deterministic graders for each ADEM task.
Each grader returns a score in [0.0, 1.0].
"""
from __future__ import annotations
from typing import Dict
from .env import ADEMEnvironment


class ADEMGrader:
    """Wraps ADEMEnvironment.compute_final_score() with per-task grading logic."""

    @staticmethod
    def grade(env: ADEMEnvironment) -> Dict[str, float]:
        """
        Grade the completed episode.
        Returns dict with 'score' (0.0–1.0) and sub-metric breakdown.
        """
        result = env.compute_final_score()

        # Task-specific post-processing
        if env.task_name == "controlled_evacuation":
            # Easy task: penalize less for congestion, more for pure survival
            adjusted = (
                0.45 * result["survival_rate"]
                + 0.25 * result["evacuation_efficiency"]
                + 0.15 * result["shelter_balance"]
                + 0.08 * result["congestion_score"]
                + 0.07 * result["safety_score"]
            )
            result["score"] = round(min(1.0, max(0.0, adjusted)), 4)

        elif env.task_name == "dynamic_hazard":
            # Medium: evacuation speed matters more (hazard spreading)
            adjusted = (
                0.35 * result["survival_rate"]
                + 0.25 * result["evacuation_efficiency"]
                + 0.15 * result["shelter_balance"]
                + 0.15 * result["congestion_score"]
                + 0.10 * result["safety_score"]
            )
            result["score"] = round(min(1.0, max(0.0, adjusted)), 4)

        elif env.task_name == "panic_evacuation":
            # Hard: safety + survival most critical under panic/dual hazards
            adjusted = (
                0.35 * result["survival_rate"]
                + 0.20 * result["evacuation_efficiency"]
                + 0.15 * result["shelter_balance"]
                + 0.15 * result["congestion_score"]
                + 0.15 * result["safety_score"]
            )
            result["score"] = round(min(1.0, max(0.0, adjusted)), 4)

        return result