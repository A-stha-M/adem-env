"""
Core ADEM simulation environment.
Implements the OpenEnv step() / reset() / state() interface.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .models import ADEMAction, ADEMObservation, ADEMReward
from .tasks import TASKS


def _to_int(v) -> int:
    """Convert any numeric type (including numpy) to Python int."""
    return int(v)


def _to_float(v) -> float:
    """Convert any numeric type (including numpy) to Python float."""
    return float(v)


def _grid_to_int_list(arr: np.ndarray):
    """Convert NxN numpy int array to nested Python list of ints."""
    return [[int(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]


def _grid_to_float_list(arr: np.ndarray, decimals: int = 3):
    """Convert NxN numpy float array to nested Python list of floats."""
    return [[round(float(arr[i][j]), decimals) for j in range(arr.shape[1])] for i in range(arr.shape[0])]


class ADEMEnvironment:
    """
    Adaptive Disaster Evacuation Management environment.
    Grid-based city simulation where agent issues evacuation orders.
    """

    DIRECTIONS = {
        "N": (-1, 0),
        "S": (1, 0),
        "E": (0, 1),
        "W": (0, -1),
        "STAY": (0, 0),
    }

    def __init__(self, task: str = "controlled_evacuation", seed: Optional[int] = None):
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASKS.keys())}")
        self.task_name = task
        self.cfg = copy.deepcopy(TASKS[task])
        self._seed = seed if seed is not None else self.cfg["seed"]

        self.n: int = self.cfg["grid_size"]
        self.population_grid: np.ndarray = np.zeros((self.n, self.n), dtype=np.int32)
        self.hazard_grid: np.ndarray = np.zeros((self.n, self.n), dtype=np.float64)
        self.road_blockages: np.ndarray = np.zeros((self.n, self.n), dtype=np.int32)
        self.congestion_levels: np.ndarray = np.zeros((self.n, self.n), dtype=np.float64)
        self.panic_levels: np.ndarray = np.zeros((self.n, self.n), dtype=np.float64)
        self.vulnerable_pop: np.ndarray = np.zeros((self.n, self.n), dtype=np.int32)

        self.shelter_positions: Dict[str, Tuple[int, int]] = {}
        self.shelter_capacities: Dict[str, int] = {}
        self.shelter_populations: Dict[str, int] = {}
        self.available_resources: Dict[str, int] = {}

        self.current_step: int = 0
        self.max_steps: int = self.cfg["max_steps"]
        self.total_initial_pop: int = 0
        self.evacuated: int = 0
        self.casualties: int = 0
        self.episode_rewards: list = []

        self._rng = random.Random(self._seed)
        self._np_rng = np.random.RandomState(self._seed)

    # ──────────────────────────────────────────────
    # Public OpenEnv interface
    # ──────────────────────────────────────────────

    def reset(self) -> ADEMObservation:
        """Reset environment to initial state, return first observation."""
        self._rng = random.Random(self._seed)
        self._np_rng = np.random.RandomState(self._seed)
        n = self.n

        self.population_grid = np.zeros((n, n), dtype=np.int32)
        self.hazard_grid = np.zeros((n, n), dtype=np.float64)
        self.road_blockages = np.zeros((n, n), dtype=np.int32)
        self.congestion_levels = np.zeros((n, n), dtype=np.float64)
        self.panic_levels = np.zeros((n, n), dtype=np.float64)
        self.vulnerable_pop = np.zeros((n, n), dtype=np.int32)

        # Place population clusters
        for center in self.cfg["population_centers"]:
            cx, cy = center["pos"]
            total = center["count"]
            radius = center.get("radius", 1)
            cells = [
                (cx + dx, cy + dy)
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
                if 0 <= cx + dx < n and 0 <= cy + dy < n
            ]
            per_cell = max(1, total // len(cells))
            for rx, ry in cells:
                self.population_grid[rx][ry] += per_cell
                self.vulnerable_pop[rx][ry] += max(0, per_cell // 5)

        # Place hazards
        for h in self.cfg["hazards"]:
            hx, hy = h["pos"]
            self.hazard_grid[hx][hy] = float(h["intensity"])

        # Setup shelters
        self.shelter_positions = {}
        self.shelter_capacities = {}
        self.shelter_populations = {}
        for s in self.cfg["shelters"]:
            sid = s["id"]
            pos = (int(s["pos"][0]), int(s["pos"][1]))
            self.shelter_positions[sid] = pos
            self.shelter_capacities[sid] = int(s["capacity"])
            self.shelter_populations[sid] = 0
            self.population_grid[pos[0]][pos[1]] = 0
            self.vulnerable_pop[pos[0]][pos[1]] = 0

        # Resources — all Python ints
        self.available_resources = {k: int(v) for k, v in self.cfg.get("resources", {"drones": 2, "ambulances": 1}).items()}

        self.total_initial_pop = int(self.population_grid.sum())
        self.evacuated = 0
        self.casualties = 0
        self.current_step = 0
        self.episode_rewards = []

        if self.cfg.get("panic_enabled", False):
            for i in range(n):
                for j in range(n):
                    if self.population_grid[i][j] > 0:
                        self.panic_levels[i][j] = float(self._np_rng.uniform(0.05, 0.30))

        return self._build_observation()

    def step(self, action: ADEMAction) -> Tuple[ADEMObservation, float, bool, Dict[str, Any]]:
        """Advance environment one timestep. Returns (observation, reward, done, info)."""
        self.current_step += 1
        n = self.n

        # 1. Apply road controls
        for road_key, is_open in action.road_controls.items():
            try:
                rx, ry = self._parse_key(road_key)
                if self._in_bounds(rx, ry):
                    self.road_blockages[rx][ry] = 0 if is_open else 1
            except (ValueError, IndexError):
                pass

        # 2. Deploy drone (reduces panic)
        if action.deploy_drone and self.available_resources.get("drones", 0) > 0:
            try:
                dx, dy = self._parse_key(action.deploy_drone)
                if self._in_bounds(dx, dy):
                    self.panic_levels[dx][dy] = max(0.0, float(self.panic_levels[dx][dy]) - 0.30)
                    self.available_resources["drones"] = int(self.available_resources["drones"]) - 1
            except (ValueError, IndexError):
                pass

        # 3. Move population
        evacuated_step, casualties_step = self._apply_movement(action.zone_directions)

        # 4. Spread hazard
        self._spread_hazard()

        # 5. Apply hazard damage
        step_casualties = self._apply_hazard_damage()
        casualties_step += step_casualties

        # 6. Update congestion
        self._update_congestion()

        # 7. Update panic (hard task only)
        if self.cfg.get("panic_enabled", False):
            self._update_panic()

        # 8. Compute reward
        reward = self._compute_step_reward(evacuated_step, casualties_step)
        self.episode_rewards.append(reward)

        # 9. Done condition
        remaining = int(self.population_grid.sum())
        done = bool(
            remaining == 0
            or self.current_step >= self.max_steps
            or self.casualties >= int(self.total_initial_pop * 0.6)
        )

        # All info values explicitly cast to Python native types
        info: Dict[str, Any] = {
            "step": int(self.current_step),
            "evacuated": int(self.evacuated),
            "casualties": int(self.casualties),
            "remaining": int(remaining),
            "evacuated_this_step": int(evacuated_step),
            "casualties_this_step": int(casualties_step),
        }

        return self._build_observation(), float(reward), done, info

    def state(self) -> Dict[str, Any]:
        """Return current episode state summary."""
        remaining = int(self.population_grid.sum())
        return {
            "task": str(self.task_name),
            "step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "total_initial_population": int(self.total_initial_pop),
            "evacuated": int(self.evacuated),
            "casualties": int(self.casualties),
            "remaining": int(remaining),
            "hazard_cells": int((self.hazard_grid > 0.5).sum()),
            "blocked_roads": int(self.road_blockages.sum()),
            "shelter_populations": {k: int(v) for k, v in self.shelter_populations.items()},
        }

    def compute_final_score(self) -> Dict[str, float]:
        """Deterministic grader — produces score in [0.0, 1.0]."""
        n = self.n
        weights = self.cfg["score_weights"]

        survival_rate = float(max(0.0, (self.total_initial_pop - self.casualties))) / float(max(1, self.total_initial_pop))

        evac_fraction = float(self.evacuated) / float(max(1, self.total_initial_pop))
        time_bonus = 1.0 - (float(self.current_step) / float(self.max_steps)) * 0.5
        evacuation_efficiency = min(1.0, evac_fraction * time_bonus)

        shelter_balance = self._shelter_balance_score()

        congestion_score = max(0.0, 1.0 - float(self.congestion_levels.mean()))

        high_risk_pop = int(sum(
            self.population_grid[i][j]
            for i in range(n)
            for j in range(n)
            if self.hazard_grid[i][j] > 0.5
        ))
        safety_score = max(0.0, 1.0 - float(high_risk_pop) / float(max(1, self.total_initial_pop)))

        final = (
            float(weights["survival"]) * survival_rate
            + float(weights["evacuation_efficiency"]) * evacuation_efficiency
            + float(weights["shelter_balance"]) * shelter_balance
            + float(weights["congestion"]) * congestion_score
            + float(weights["safety"]) * safety_score
        )
        final = float(min(1.0, max(0.0, final)))

        return {
            "score": round(final, 4),
            "survival_rate": round(survival_rate, 4),
            "evacuation_efficiency": round(evacuation_efficiency, 4),
            "shelter_balance": round(shelter_balance, 4),
            "congestion_score": round(congestion_score, 4),
            "safety_score": round(safety_score, 4),
        }

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _apply_movement(self, zone_directions: Dict[str, str]) -> Tuple[int, int]:
        """Move population according to agent directions. Returns (evacuated, casualties)."""
        n = self.n
        new_pop = self.population_grid.copy()
        new_vuln = self.vulnerable_pop.copy()
        evacuated_step = 0
        casualties_step = 0

        for zone_key, direction in zone_directions.items():
            direction = direction.upper().strip()
            if direction not in self.DIRECTIONS:
                continue
            try:
                zx, zy = self._parse_key(zone_key)
            except ValueError:
                continue
            if not self._in_bounds(zx, zy):
                continue

            pop = int(self.population_grid[zx][zy])
            vuln = int(self.vulnerable_pop[zx][zy])
            if pop == 0:
                continue
            if direction == "STAY":
                continue

            panic = float(self.panic_levels[zx][zy])
            compliance = 1.0 - panic * 0.65
            moving_pop = max(1, int(pop * compliance))
            moving_pop = min(moving_pop, max(1, pop // 2))
            moving_vuln = min(vuln, moving_pop // 5)

            ddx, ddy = self.DIRECTIONS[direction]
            nx, ny = zx + ddx, zy + ddy

            if not self._in_bounds(nx, ny):
                continue
            if int(self.road_blockages[nx][ny]) == 1:
                continue

            shelter_id = self._shelter_at(nx, ny)
            if shelter_id is not None:
                cap = int(self.shelter_capacities[shelter_id])
                cur = int(self.shelter_populations[shelter_id])
                space = max(0, cap - cur)
                accepted = min(moving_pop, space)
                if accepted > 0:
                    self.shelter_populations[shelter_id] = cur + accepted
                    new_pop[zx][zy] -= accepted
                    new_vuln[zx][zy] -= min(moving_vuln, accepted // 5 + 1)
                    self.evacuated += accepted
                    evacuated_step += accepted
            else:
                new_pop[zx][zy] -= moving_pop
                new_pop[nx][ny] += moving_pop
                new_vuln[zx][zy] -= moving_vuln
                new_vuln[nx][ny] += moving_vuln

        self.population_grid = np.clip(new_pop, 0, None).astype(np.int32)
        self.vulnerable_pop = np.clip(new_vuln, 0, None).astype(np.int32)
        return int(evacuated_step), int(casualties_step)

    def _spread_hazard(self):
        """Spread hazard to adjacent cells."""
        spread_rate = float(self.cfg.get("hazard_spread_rate", 0.0))
        if spread_rate == 0.0:
            return
        n = self.n
        new_hazard = self.hazard_grid.copy()
        for i in range(n):
            for j in range(n):
                if self.hazard_grid[i][j] > 0.3:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if self._in_bounds(ni, nj):
                            increment = float(self.hazard_grid[i][j]) * spread_rate
                            new_hazard[ni][nj] = min(1.0, float(new_hazard[ni][nj]) + increment)
                    if self.hazard_grid[i][j] > 0.85:
                        self.road_blockages[i][j] = 1
        self.hazard_grid = new_hazard

    def _apply_hazard_damage(self) -> int:
        """Kill fraction of people exposed to high hazard. Returns casualties count."""
        n = self.n
        step_casualties = 0
        for i in range(n):
            for j in range(n):
                pop = int(self.population_grid[i][j])
                hz = float(self.hazard_grid[i][j])
                if hz > 0.65 and pop > 0:
                    cas = max(1, int(pop * (hz - 0.5) * 0.12))
                    cas = min(cas, pop)
                    self.population_grid[i][j] = int(self.population_grid[i][j]) - cas
                    self.vulnerable_pop[i][j] = max(0, int(self.vulnerable_pop[i][j]) - cas // 5)
                    self.casualties += cas
                    step_casualties += cas
        return int(step_casualties)

    def _update_congestion(self):
        """Recompute congestion from population density."""
        max_density = 30.0
        self.congestion_levels = np.clip(
            self.population_grid.astype(np.float64) / max_density, 0.0, 1.0
        )

    def _update_panic(self):
        """Panic grows with nearby hazard and congestion (hard task)."""
        n = self.n
        for i in range(n):
            for j in range(n):
                if self.population_grid[i][j] > 0:
                    hz_influence = float(self.hazard_grid[i][j]) * 0.25
                    cong_influence = float(self.congestion_levels[i][j]) * 0.15
                    self.panic_levels[i][j] = min(
                        1.0, float(self.panic_levels[i][j]) + hz_influence + cong_influence
                    )

    def _compute_step_reward(self, evacuated_step: int, casualties_step: int) -> float:
        """Dense per-step reward signal."""
        n = self.n

        evac_reward = 0.05 * float(evacuated_step)

        total_vuln_saved = float(sum(self.shelter_populations.values()))
        vuln_reward = 0.02 * (total_vuln_saved / float(max(1, self.total_initial_pop))) * float(self.n)

        cas_penalty = -0.08 * float(casualties_step)

        avg_congestion = float(self.congestion_levels.mean())
        cong_penalty = -0.03 * avg_congestion * 10.0

        high_risk_pop = float(sum(
            int(self.population_grid[i][j])
            for i in range(n)
            for j in range(n)
            if self.hazard_grid[i][j] > 0.5
        ))
        risk_penalty = -0.04 * (high_risk_pop / float(max(1, self.total_initial_pop))) * 10.0

        shelter_bal = 0.01 * self._shelter_balance_score()

        return float(evac_reward + vuln_reward + cas_penalty + cong_penalty + risk_penalty + shelter_bal)

    def _shelter_balance_score(self) -> float:
        """Score [0,1] reflecting how evenly shelters are utilized."""
        if not self.shelter_populations:
            return 0.0
        utils = [
            float(self.shelter_populations[sid]) / float(max(1, self.shelter_capacities[sid]))
            for sid in self.shelter_populations
        ]
        if len(utils) < 2:
            return 1.0
        avg = sum(utils) / len(utils)
        variance = sum((u - avg) ** 2 for u in utils) / len(utils)
        return float(max(0.0, 1.0 - variance * 5.0))

    def _shelter_at(self, row: int, col: int) -> Optional[str]:
        for sid, pos in self.shelter_positions.items():
            if pos[0] == row and pos[1] == col:
                return sid
        return None

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    @staticmethod
    def _parse_key(key: str) -> Tuple[int, int]:
        parts = key.split(",")
        return int(parts[0].strip()), int(parts[1].strip())

    def _build_observation(self) -> ADEMObservation:
        """Build observation with all types explicitly cast to Python natives."""
        return ADEMObservation(
            population_grid=_grid_to_int_list(self.population_grid),
            hazard_grid=_grid_to_float_list(self.hazard_grid),
            road_blockages=_grid_to_int_list(self.road_blockages),
            shelter_capacities={k: int(v) for k, v in self.shelter_capacities.items()},
            shelter_populations={k: int(v) for k, v in self.shelter_populations.items()},
            congestion_levels=_grid_to_float_list(self.congestion_levels),
            panic_levels=_grid_to_float_list(self.panic_levels),
            vulnerable_population_map=_grid_to_int_list(self.vulnerable_pop),
            available_resources={k: int(v) for k, v in self.available_resources.items()},
            time_remaining=int(self.max_steps - self.current_step),
            total_population=int(self.total_initial_pop),
            evacuated=int(self.evacuated),
            casualties=int(self.casualties),
            grid_size=int(self.n),
        )