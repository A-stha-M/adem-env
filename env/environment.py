"""
Core ADEM simulation environment.
Implements the OpenEnv step() / reset() / state() interface.

New mechanics vs v1:
  - wind_direction: biases hazard spread in one compass direction
  - initial_road_blockages: pre-blocked roads at episode start
  - aftershock_probability: stochastic road blocking each step
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import ADEMAction, ADEMObservation
from tasks import TASKS

# ─── Grid serialisation helpers ────────────────────────────────────────────────

def _g_int(arr: np.ndarray) -> List[List[int]]:
    return [[int(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]


def _g_float(arr: np.ndarray, dec: int = 3) -> List[List[float]]:
    return [[round(float(arr[i][j]), dec) for j in range(arr.shape[1])] for i in range(arr.shape[0])]


# ─── Direction mapping ──────────────────────────────────────────────────────────

DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1), "STAY": (0, 0),
}

# Wind multipliers per spread direction given wind_direction config
# Key = (wind_direction, spread_direction), Value = multiplier
_WIND_MULT: Dict[Tuple[str, str], float] = {
    ("N", "N"): 2.5, ("N", "S"): 0.2, ("N", "E"): 0.7, ("N", "W"): 0.7,
    ("S", "S"): 2.5, ("S", "N"): 0.2, ("S", "E"): 0.7, ("S", "W"): 0.7,
    ("E", "E"): 2.5, ("E", "W"): 0.2, ("E", "N"): 0.7, ("E", "S"): 0.7,
    ("W", "W"): 2.5, ("W", "E"): 0.2, ("W", "N"): 0.7, ("W", "S"): 0.7,
}


class ADEMEnvironment:
    """
    Adaptive Disaster Evacuation Management — grid-based city simulation.

    The agent is the central command system. Each timestep it issues:
      - zone_directions: move civilians in each occupied cell
      - road_controls:   open or close road segments
      - resource_allocations: send resources to shelters
      - deploy_drone:    reduce panic in a zone

    Supports 9 task configurations with escalating complexity.
    """

    def __init__(self, task: str = "controlled_evacuation", seed: Optional[int] = None):
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {list(TASKS.keys())}")
        self.task_name = task
        self.cfg = copy.deepcopy(TASKS[task])
        self._seed = seed if seed is not None else int(self.cfg["seed"])
        self.n: int = int(self.cfg["grid_size"])
        self.max_steps: int = int(self.cfg["max_steps"])

        # Grids — allocated in reset()
        self.population_grid = np.zeros((self.n, self.n), dtype=np.int32)
        self.hazard_grid = np.zeros((self.n, self.n), dtype=np.float64)
        self.road_blockages = np.zeros((self.n, self.n), dtype=np.int32)
        self.congestion_levels = np.zeros((self.n, self.n), dtype=np.float64)
        self.panic_levels = np.zeros((self.n, self.n), dtype=np.float64)
        self.vulnerable_pop = np.zeros((self.n, self.n), dtype=np.int32)

        # Shelter state
        self.shelter_positions: Dict[str, Tuple[int, int]] = {}
        self.shelter_capacities: Dict[str, int] = {}
        self.shelter_populations: Dict[str, int] = {}

        # Resource pool
        self.available_resources: Dict[str, int] = {}

        # Episode counters
        self.current_step: int = 0
        self.total_initial_pop: int = 0
        self.evacuated: int = 0
        self.casualties: int = 0
        self.episode_rewards: List[float] = []

        self._rng = random.Random(self._seed)
        self._np_rng = np.random.RandomState(self._seed)

    # ── Public OpenEnv Interface ──────────────────────────────────────────────

    def reset(self) -> ADEMObservation:
        """Reset to clean initial state and return first observation."""
        self._rng = random.Random(self._seed)
        self._np_rng = np.random.RandomState(self._seed)
        n = self.n

        # Clear grids
        self.population_grid = np.zeros((n, n), dtype=np.int32)
        self.hazard_grid = np.zeros((n, n), dtype=np.float64)
        self.road_blockages = np.zeros((n, n), dtype=np.int32)
        self.congestion_levels = np.zeros((n, n), dtype=np.float64)
        self.panic_levels = np.zeros((n, n), dtype=np.float64)
        self.vulnerable_pop = np.zeros((n, n), dtype=np.int32)

        # Place population clusters
        for center in self.cfg["population_centers"]:
            cx, cy = int(center["pos"][0]), int(center["pos"][1])
            total = int(center["count"])
            radius = int(center.get("radius", 1))
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
            hx, hy = int(h["pos"][0]), int(h["pos"][1])
            self.hazard_grid[hx][hy] = float(h["intensity"])

        # Setup shelters — clear their cells of initial population
        self.shelter_positions = {}
        self.shelter_capacities = {}
        self.shelter_populations = {}
        for s in self.cfg["shelters"]:
            sid = str(s["id"])
            pos = (int(s["pos"][0]), int(s["pos"][1]))
            self.shelter_positions[sid] = pos
            self.shelter_capacities[sid] = int(s["capacity"])
            self.shelter_populations[sid] = 0
            self.population_grid[pos[0]][pos[1]] = 0
            self.vulnerable_pop[pos[0]][pos[1]] = 0

        # Resources
        self.available_resources = {
            k: int(v) for k, v in self.cfg.get("resources", {"drones": 2, "ambulances": 1}).items()
        }

        # Apply pre-episode road blockages (earthquake damage, coastal flooding)
        for block in self.cfg.get("initial_road_blockages", []):
            bx, by = int(block[0]), int(block[1])
            if self._in_bounds(bx, by) and self._shelter_at(bx, by) is None:
                self.road_blockages[bx][by] = 1

        # Episode counters
        self.total_initial_pop = int(self.population_grid.sum())
        self.evacuated = 0
        self.casualties = 0
        self.current_step = 0
        self.episode_rewards = []

        # Initial panic (panic-enabled tasks start with some crowd anxiety)
        if self.cfg.get("panic_enabled", False):
            for i in range(n):
                for j in range(n):
                    if self.population_grid[i][j] > 0:
                        self.panic_levels[i][j] = float(
                            self._np_rng.uniform(0.05, 0.30)
                        )

        return self._build_observation()

    def step(self, action: ADEMAction) -> Tuple[ADEMObservation, float, bool, Dict[str, Any]]:
        """Advance one timestep. Returns (observation, reward, done, info)."""
        self.current_step += 1

        # 1. Apply road controls (agent can open/close roads)
        for road_key, is_open in action.road_controls.items():
            try:
                rx, ry = self._parse_key(road_key)
                if self._in_bounds(rx, ry):
                    self.road_blockages[rx][ry] = 0 if is_open else 1
            except (ValueError, IndexError):
                pass

        # 2. Deploy drone → reduce panic in target zone
        if action.deploy_drone and self.available_resources.get("drones", 0) > 0:
            try:
                dx, dy = self._parse_key(action.deploy_drone)
                if self._in_bounds(dx, dy):
                    self.panic_levels[dx][dy] = max(
                        0.0, float(self.panic_levels[dx][dy]) - 0.35
                    )
                    # Neighbouring cells also calmed slightly
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = dx + di, dy + dj
                        if self._in_bounds(ni, nj):
                            self.panic_levels[ni][nj] = max(
                                0.0, float(self.panic_levels[ni][nj]) - 0.10
                            )
                    self.available_resources["drones"] = (
                        int(self.available_resources["drones"]) - 1
                    )
            except (ValueError, IndexError):
                pass

        # 3. Move population per zone directions
        evacuated_step, casualties_step = self._apply_movement(action.zone_directions)

        # 4. Spread hazard (with optional wind bias)
        self._spread_hazard()

        # 5. Aftershocks — stochastically block new roads
        if self.cfg.get("aftershock_probability", 0.0) > 0:
            self._apply_aftershocks()

        # 6. Hazard damage — casualties from high-intensity zones
        step_casualties = self._apply_hazard_damage()
        casualties_step += step_casualties

        # 7. Update congestion
        self._update_congestion()

        # 8. Update panic (panic-enabled tasks only)
        if self.cfg.get("panic_enabled", False):
            self._update_panic()

        # 9. Compute dense per-step reward
        reward = self._compute_step_reward(evacuated_step, casualties_step)
        self.episode_rewards.append(reward)

        # 10. Episode termination
        remaining = int(self.population_grid.sum())
        done = bool(
            remaining == 0
            or self.current_step >= self.max_steps
            or self.casualties >= int(self.total_initial_pop * 0.65)
        )

        info: Dict[str, Any] = {
            "step": int(self.current_step),
            "evacuated": int(self.evacuated),
            "casualties": int(self.casualties),
            "remaining": int(remaining),
            "evacuated_this_step": int(evacuated_step),
            "casualties_this_step": int(casualties_step),
            "task": str(self.task_name),
        }

        return self._build_observation(), float(reward), done, info

    def state(self) -> Dict[str, Any]:
        """Return current episode state summary (OpenEnv /state endpoint)."""
        remaining = int(self.population_grid.sum())
        return {
            "task": str(self.task_name),
            "difficulty": str(self.cfg.get("difficulty", "unknown")),
            "step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "total_initial_population": int(self.total_initial_pop),
            "evacuated": int(self.evacuated),
            "casualties": int(self.casualties),
            "remaining": int(remaining),
            "survival_rate": round(
                float(self.total_initial_pop - self.casualties) / max(1, self.total_initial_pop), 4
            ),
            "hazard_cells_active": int((self.hazard_grid > 0.5).sum()),
            "blocked_roads": int(self.road_blockages.sum()),
            "shelter_populations": {k: int(v) for k, v in self.shelter_populations.items()},
            "shelter_capacities": {k: int(v) for k, v in self.shelter_capacities.items()},
            "available_resources": {k: int(v) for k, v in self.available_resources.items()},
            "wind_direction": str(self.cfg.get("wind_direction") or "none"),
            "panic_enabled": bool(self.cfg.get("panic_enabled", False)),
        }

    def compute_final_score(self) -> Dict[str, float]:
        """
        Deterministic grader — score in [0.0, 1.0].
        All sub-metrics are independently normalised before weighting.
        """
        n = self.n
        weights = self.cfg["score_weights"]

        # 1. Survival rate
        survival_rate = float(
            max(0, self.total_initial_pop - self.casualties)
        ) / float(max(1, self.total_initial_pop))

        # 2. Evacuation efficiency (fraction × time bonus)
        evac_fraction = float(self.evacuated) / float(max(1, self.total_initial_pop))
        time_bonus = 1.0 - (float(self.current_step) / float(self.max_steps)) * 0.5
        evacuation_efficiency = min(1.0, evac_fraction * time_bonus)

        # 3. Shelter utilisation balance (low variance = high score)
        shelter_balance = self._shelter_balance_score()

        # 4. Congestion score
        congestion_score = max(0.0, 1.0 - float(self.congestion_levels.mean()))

        # 5. Safety score (people NOT in high-risk zones)
        high_risk_pop = int(sum(
            self.population_grid[i][j]
            for i in range(n) for j in range(n)
            if self.hazard_grid[i][j] > 0.5
        ))
        safety_score = max(
            0.0,
            1.0 - float(high_risk_pop) / float(max(1, self.total_initial_pop)),
        )

        final = float(
            weights["survival"] * survival_rate
            + weights["evacuation_efficiency"] * evacuation_efficiency
            + weights["shelter_balance"] * shelter_balance
            + weights["congestion"] * congestion_score
            + weights["safety"] * safety_score
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

    # ── Internal Simulation Methods ───────────────────────────────────────────

    def _apply_movement(self, zone_directions: Dict[str, str]) -> Tuple[int, int]:
        """Move population per agent directions. Returns (evacuated, casualties)."""
        n = self.n
        new_pop = self.population_grid.copy()
        new_vuln = self.vulnerable_pop.copy()
        evacuated_step = 0
        casualties_step = 0

        for zone_key, direction in zone_directions.items():
            direction = direction.upper().strip()
            if direction not in DIRECTION_DELTAS:
                continue
            try:
                zx, zy = self._parse_key(zone_key)
            except ValueError:
                continue
            if not self._in_bounds(zx, zy):
                continue

            pop = int(self.population_grid[zx][zy])
            vuln = int(self.vulnerable_pop[zx][zy])
            if pop == 0 or direction == "STAY":
                continue

            # Panic reduces civilian compliance with agent orders
            panic = float(self.panic_levels[zx][zy])
            compliance = max(0.1, 1.0 - panic * 0.65)
            moving_pop = max(1, int(pop * compliance))
            moving_pop = min(moving_pop, max(1, pop // 2))  # realistic crowd speed
            moving_vuln = min(vuln, max(0, moving_pop // 5))

            ddx, ddy = DIRECTION_DELTAS[direction]
            nx, ny = zx + ddx, zy + ddy

            if not self._in_bounds(nx, ny):
                continue
            if int(self.road_blockages[nx][ny]) == 1:
                continue  # Road blocked — civilians can't pass

            shelter_id = self._shelter_at(nx, ny)
            if shelter_id is not None:
                # Destination is a shelter — evacuate civilians
                cap = int(self.shelter_capacities[shelter_id])
                cur = int(self.shelter_populations[shelter_id])
                space = max(0, cap - cur)
                accepted = min(moving_pop, space)
                if accepted > 0:
                    self.shelter_populations[shelter_id] = cur + accepted
                    new_pop[zx][zy] -= accepted
                    new_vuln[zx][zy] = max(0, int(new_vuln[zx][zy]) - min(moving_vuln, accepted // 5 + 1))
                    self.evacuated += accepted
                    evacuated_step += accepted
            else:
                # Regular move toward shelter
                new_pop[zx][zy] -= moving_pop
                new_pop[nx][ny] += moving_pop
                new_vuln[zx][zy] -= moving_vuln
                new_vuln[nx][ny] += moving_vuln

        self.population_grid = np.clip(new_pop, 0, None).astype(np.int32)
        self.vulnerable_pop = np.clip(new_vuln, 0, None).astype(np.int32)
        return int(evacuated_step), int(casualties_step)

    def _spread_hazard(self):
        """
        Spread hazard to adjacent cells.
        If wind_direction is set, that direction gets 2.5× spread rate,
        the opposite direction gets 0.2× (upwind barely spreads).
        """
        spread_rate = float(self.cfg.get("hazard_spread_rate", 0.0))
        if spread_rate == 0.0:
            return

        wind = self.cfg.get("wind_direction")  # None or "N"/"S"/"E"/"W"
        n = self.n
        new_hazard = self.hazard_grid.copy()

        spread_dirs = [("N", -1, 0), ("S", 1, 0), ("E", 0, 1), ("W", 0, -1)]

        for i in range(n):
            for j in range(n):
                if self.hazard_grid[i][j] > 0.25:
                    for dir_name, di, dj in spread_dirs:
                        ni, nj = i + di, j + dj
                        if self._in_bounds(ni, nj):
                            # Wind bias
                            mult = 1.0
                            if wind:
                                mult = _WIND_MULT.get((wind, dir_name), 1.0)
                            increment = float(self.hazard_grid[i][j]) * spread_rate * mult
                            new_hazard[ni][nj] = min(
                                1.0, float(new_hazard[ni][nj]) + increment
                            )

                    # Extreme hazard (>0.85) physically blocks the road at that cell
                    if self.hazard_grid[i][j] > 0.85:
                        self.road_blockages[i][j] = 1

        self.hazard_grid = new_hazard

    def _apply_aftershocks(self):
        """
        Stochastically block additional roads near active hazard zones.
        Models earthquake aftershocks, storm intensification, gas explosions.
        """
        prob = float(self.cfg.get("aftershock_probability", 0.0))
        n = self.n
        for i in range(n):
            for j in range(n):
                if self.hazard_grid[i][j] > 0.45:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (
                            self._in_bounds(ni, nj)
                            and self._shelter_at(ni, nj) is None
                            and self._rng.random() < prob
                        ):
                            self.road_blockages[ni][nj] = 1

    def _apply_hazard_damage(self) -> int:
        """Civilians in high-hazard cells suffer casualties. Returns count."""
        n = self.n
        step_casualties = 0
        for i in range(n):
            for j in range(n):
                pop = int(self.population_grid[i][j])
                hz = float(self.hazard_grid[i][j])
                if hz > 0.65 and pop > 0:
                    # Casualty rate proportional to hazard intensity above threshold
                    cas = max(1, int(pop * (hz - 0.5) * 0.12))
                    cas = min(cas, pop)
                    self.population_grid[i][j] = int(self.population_grid[i][j]) - cas
                    self.vulnerable_pop[i][j] = max(
                        0, int(self.vulnerable_pop[i][j]) - cas // 5
                    )
                    self.casualties += cas
                    step_casualties += cas
        return int(step_casualties)

    def _update_congestion(self):
        """Congestion = normalised population density (30 people/cell = max)."""
        self.congestion_levels = np.clip(
            self.population_grid.astype(np.float64) / 30.0, 0.0, 1.0
        )

    def _update_panic(self):
        """
        Panic increases with nearby hazard intensity and local congestion.
        High congestion + high hazard = runaway panic (realistic crowd psychology).
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if self.population_grid[i][j] > 0:
                    hz_influence = float(self.hazard_grid[i][j]) * 0.22
                    cong_influence = float(self.congestion_levels[i][j]) * 0.12
                    # Cascade: congestion amplifies panic near hazard
                    cascade = hz_influence * float(self.congestion_levels[i][j]) * 0.15
                    self.panic_levels[i][j] = min(
                        1.0,
                        float(self.panic_levels[i][j]) + hz_influence + cong_influence + cascade,
                    )

    def _compute_step_reward(self, evacuated_step: int, casualties_step: int) -> float:
        """
        Dense multi-component per-step reward.
        Provides signal at every timestep (not just episode end).
        """
        n = self.n

        # Positive: people evacuated this step
        evac_r = 0.05 * float(evacuated_step)

        # Positive: cumulative shelter balance bonus (small per step)
        balance_r = 0.01 * self._shelter_balance_score()

        # Positive: vulnerable pop ratio in shelters (proxy for priority)
        total_sheltered = float(sum(self.shelter_populations.values()))
        vuln_r = 0.02 * (total_sheltered / float(max(1, self.total_initial_pop))) * float(self.n) * 0.1

        # Negative: casualties this step
        cas_p = -0.08 * float(casualties_step)

        # Negative: mean congestion (penalises bottlenecks)
        cong_p = -0.03 * float(self.congestion_levels.mean()) * 10.0

        # Negative: population still in high-risk hazard zones
        high_risk_pop = float(sum(
            int(self.population_grid[i][j])
            for i in range(n) for j in range(n)
            if self.hazard_grid[i][j] > 0.5
        ))
        risk_p = -0.04 * (high_risk_pop / float(max(1, self.total_initial_pop))) * 10.0

        # Negative: time pressure (small per-step penalty to encourage speed)
        time_p = -0.002

        return float(evac_r + balance_r + vuln_r + cas_p + cong_p + risk_p + time_p)

    def _shelter_balance_score(self) -> float:
        """
        [0, 1] — how evenly shelters are utilised.
        Perfectly equal utilisation = 1.0. High variance = 0.0.
        """
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
        """Build fully type-safe observation (all numpy types cast to Python natives)."""
        return ADEMObservation(
            population_grid=_g_int(self.population_grid),
            hazard_grid=_g_float(self.hazard_grid),
            road_blockages=_g_int(self.road_blockages),
            shelter_capacities={k: int(v) for k, v in self.shelter_capacities.items()},
            shelter_populations={k: int(v) for k, v in self.shelter_populations.items()},
            congestion_levels=_g_float(self.congestion_levels),
            panic_levels=_g_float(self.panic_levels),
            vulnerable_population_map=_g_int(self.vulnerable_pop),
            available_resources={k: int(v) for k, v in self.available_resources.items()},
            time_remaining=int(self.max_steps - self.current_step),
            total_population=int(self.total_initial_pop),
            evacuated=int(self.evacuated),
            casualties=int(self.casualties),
            grid_size=int(self.n),
        )