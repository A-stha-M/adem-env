"""
Task configurations for ADEM environment.
Three tasks from easy to hard, with seeded reproducibility.
"""
from typing import Any, Dict

TASKS: Dict[str, Dict[str, Any]] = {

    # ─────────────────────────────────────────────
    # TASK 1: Controlled Evacuation (EASY)
    # 6x6 grid, static hazard, 2 shelters
    # ─────────────────────────────────────────────
    "controlled_evacuation": {
        "grid_size": 6,
        "max_steps": 15,
        "seed": 42,
        "hazard_spread_rate": 0.0,      # hazard does NOT spread
        "panic_enabled": False,
        "population_centers": [
            {"pos": [1, 1], "count": 20, "radius": 1},
            {"pos": [4, 4], "count": 20, "radius": 0},
            {"pos": [1, 4], "count": 10, "radius": 0},
        ],
        "hazards": [
            {"pos": [0, 0], "intensity": 0.95},
        ],
        "shelters": [
            {"id": "S1", "pos": [5, 2], "capacity": 35},
            {"id": "S2", "pos": [5, 5], "capacity": 35},
        ],
        "resources": {"drones": 2, "ambulances": 1},
        "score_weights": {
            "survival": 0.40,
            "evacuation_efficiency": 0.25,
            "shelter_balance": 0.15,
            "congestion": 0.10,
            "safety": 0.10,
        },
    },

    # ─────────────────────────────────────────────
    # TASK 2: Dynamic Hazard (MEDIUM)
    # 8x8 grid, spreading fire, 3 shelters
    # ─────────────────────────────────────────────
    "dynamic_hazard": {
        "grid_size": 8,
        "max_steps": 20,
        "seed": 123,
        "hazard_spread_rate": 0.15,     # hazard spreads each step
        "panic_enabled": False,
        "population_centers": [
            {"pos": [2, 2], "count": 40, "radius": 1},
            {"pos": [5, 2], "count": 40, "radius": 1},
            {"pos": [3, 6], "count": 30, "radius": 1},
            {"pos": [6, 6], "count": 20, "radius": 0},
        ],
        "hazards": [
            {"pos": [0, 0], "intensity": 0.90},
        ],
        "shelters": [
            {"id": "S1", "pos": [7, 3], "capacity": 55},
            {"id": "S2", "pos": [7, 6], "capacity": 55},
            {"id": "S3", "pos": [4, 7], "capacity": 40},
        ],
        "resources": {"drones": 3, "ambulances": 2},
        "score_weights": {
            "survival": 0.35,
            "evacuation_efficiency": 0.20,
            "shelter_balance": 0.15,
            "congestion": 0.15,
            "safety": 0.15,
        },
    },

    # ─────────────────────────────────────────────
    # TASK 3: Panic Evacuation (HARD)
    # 10x10 grid, 2 hazard sources, panic dynamics
    # ─────────────────────────────────────────────
    "panic_evacuation": {
        "grid_size": 10,
        "max_steps": 25,
        "seed": 777,
        "hazard_spread_rate": 0.20,     # fast spreading
        "panic_enabled": True,           # population ignores instructions
        "population_centers": [
            {"pos": [2, 2], "count": 50, "radius": 1},
            {"pos": [5, 2], "count": 45, "radius": 1},
            {"pos": [2, 7], "count": 45, "radius": 1},
            {"pos": [5, 7], "count": 40, "radius": 1},
            {"pos": [7, 4], "count": 30, "radius": 1},
            {"pos": [3, 4], "count": 25, "radius": 0},
        ],
        "hazards": [
            {"pos": [0, 0], "intensity": 0.90},
            {"pos": [9, 9], "intensity": 0.80},
        ],
        "shelters": [
            {"id": "S1", "pos": [4, 4], "capacity": 80},
            {"id": "S2", "pos": [9, 1], "capacity": 70},
            {"id": "S3", "pos": [1, 9], "capacity": 70},
            {"id": "S4", "pos": [9, 8], "capacity": 60},
        ],
        "resources": {"drones": 4, "ambulances": 3},
        "score_weights": {
            "survival": 0.35,
            "evacuation_efficiency": 0.20,
            "shelter_balance": 0.15,
            "congestion": 0.15,
            "safety": 0.15,
        },
    },
}