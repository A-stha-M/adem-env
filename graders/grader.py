"""
Deterministic graders for each ADEM task.
Each grader returns a score in [0.0, 1.0] with a task-specific bonus component.
"""
from __future__ import annotations
from typing import Dict
from env import ADEMEnvironment


class ADEMGrader:
    """
    Wraps ADEMEnvironment.compute_final_score() with per-task grading logic.
    Each task applies a different weight profile reflecting its real-world priority.
    """

    # Task-specific weight overrides (survival, efficiency, balance, congestion, safety)
    _TASK_WEIGHTS = {
        "controlled_evacuation":  (0.45, 0.25, 0.15, 0.08, 0.07),
        "flash_flood":            (0.50, 0.25, 0.08, 0.10, 0.07),  # Pure survival + speed
        "building_fire":          (0.40, 0.30, 0.10, 0.10, 0.10),  # Efficiency matters: wind
        "dynamic_hazard":         (0.35, 0.25, 0.15, 0.15, 0.10),
        "earthquake_response":    (0.35, 0.20, 0.15, 0.20, 0.10),  # Congestion high (blocked)
        "industrial_chemical":    (0.35, 0.20, 0.15, 0.15, 0.15),
        "panic_evacuation":       (0.35, 0.20, 0.15, 0.15, 0.15),
        "hurricane_coastal":      (0.38, 0.20, 0.12, 0.15, 0.15),
        "multi_hazard_city":      (0.35, 0.20, 0.15, 0.15, 0.15),
    }

    @staticmethod
    def grade(env: ADEMEnvironment) -> Dict[str, float]:
        """
        Grade the completed episode.
        Returns dict with 'score' (0.0–1.0) and full sub-metric breakdown.
        """
        result = env.compute_final_score()
        task = env.task_name

        # Apply task-specific weights
        weights = ADEMGrader._TASK_WEIGHTS.get(task, (0.35, 0.20, 0.15, 0.15, 0.15))
        w_surv, w_evac, w_bal, w_cong, w_safe = weights

        adjusted = (
            w_surv * result["survival_rate"]
            + w_evac * result["evacuation_efficiency"]
            + w_bal  * result["shelter_balance"]
            + w_cong * result["congestion_score"]
            + w_safe * result["safety_score"]
        )

        # Task-specific bonus mechanics
        bonus = ADEMGrader._task_bonus(env, result)
        adjusted = adjusted + bonus

        result["score"] = round(float(min(1.0, max(0.0, adjusted))), 4)
        result["task_bonus"] = round(float(bonus), 4)
        result["task"] = task
        result["difficulty"] = str(env.cfg.get("difficulty", "unknown"))
        return result

    @staticmethod
    def _task_bonus(env: ADEMEnvironment, metrics: Dict[str, float]) -> float:
        """
        Small bonus (<= 0.05) for task-specific excellence:
        - flash_flood: early evacuators (before flood fully spreads)
        - earthquake: successfully opened blocked roads
        - hurricane_coastal: evacuated from the most dangerous coastal strip
        - multi_hazard_city: all 3 hazard zones have 0 remaining population
        """
        task = env.task_name
        n = env.n
        bonus = 0.0

        if task == "flash_flood":
            # Bonus if most people evacuated before step 10 (flood races fast)
            steps_used = env.current_step
            if env.evacuated > int(env.total_initial_pop * 0.6) and steps_used <= 10:
                bonus += 0.04

        elif task == "earthquake_response":
            # Bonus for road-clearing: fewer blocked roads than initially placed
            initial_blocks = len(env.cfg.get("initial_road_blockages", []))
            current_blocks = int(env.road_blockages.sum())
            if initial_blocks > 0 and current_blocks < initial_blocks:
                cleared_ratio = (initial_blocks - current_blocks) / initial_blocks
                bonus += 0.03 * cleared_ratio

        elif task == "hurricane_coastal":
            # Bonus for clearing the coastal strip (rows 7-9) of population
            coastal_remaining = int(sum(
                env.population_grid[r][c]
                for r in range(max(0, n - 3), n)
                for c in range(n)
            ))
            if coastal_remaining == 0:
                bonus += 0.04
            elif coastal_remaining < int(env.total_initial_pop * 0.05):
                bonus += 0.02

        elif task == "multi_hazard_city":
            # Bonus for clearing all 3 hazard origin zones of population
            hazard_origins = [(0, 0), (n - 1, n - 1), (0, n - 1)]
            cleared = sum(
                1 for (hr, hc) in hazard_origins
                if int(sum(
                    env.population_grid[max(0, hr - 2):min(n, hr + 3), max(0, hc - 2):min(n, hc + 3)].flatten()
                )) == 0
            )
            bonus += 0.015 * cleared  # up to 0.045 for clearing all 3

        return float(bonus)