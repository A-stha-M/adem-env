"""Task config: flash_flood (easy)."""

TASK = {
    "grid_size": 6,
    "max_steps": 15,
    "seed": 99,
    "hazard_spread_rate": 0.28,
    "wind_direction": "S",
    "panic_enabled": False,
    "initial_road_blockages": [],
    "aftershock_probability": 0.0,
    "population_centers": [
        {"pos": [0, 2], "count": 15, "radius": 0},
        {"pos": [0, 4], "count": 15, "radius": 0},
        {"pos": [1, 1], "count": 12, "radius": 0},
        {"pos": [2, 3], "count": 12, "radius": 0},
        {"pos": [2, 5], "count": 10, "radius": 0},
    ],
    "hazards": [
        {"pos": [0, 0], "intensity": 0.82},
        {"pos": [0, 3], "intensity": 0.78},
    ],
    "shelters": [
        {"id": "S1", "pos": [5, 1], "capacity": 30},
        {"id": "S2", "pos": [5, 4], "capacity": 30},
        {"id": "S3", "pos": [4, 5], "capacity": 20},
    ],
    "resources": {"drones": 1, "ambulances": 1},
    "score_weights": {
        "survival": 0.45,
        "evacuation_efficiency": 0.25,
        "shelter_balance": 0.10,
        "congestion": 0.10,
        "safety": 0.10,
    },
    "difficulty": "easy",
    "description": (
        "6x6 low-lying district beside a river. Floodwater rises from the north edge "
        "and spreads south rapidly. 64 civilians must reach 3 elevated shelters before "
        "escape routes inundate. Speed is critical."
    ),
}
