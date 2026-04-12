"""Task config: hurricane_coastal (hard)."""

TASK = {
    "grid_size": 10,
    "max_steps": 25,
    "seed": 888,
    "hazard_spread_rate": 0.22,
    "wind_direction": "N",
    "panic_enabled": True,
    "initial_road_blockages": [
        [9, 3], [9, 6], [8, 4], [8, 7],
    ],
    "aftershock_probability": 0.12,
    "population_centers": [
        {"pos": [8, 2], "count": 55, "radius": 0},
        {"pos": [8, 7], "count": 50, "radius": 0},
        {"pos": [7, 4], "count": 40, "radius": 0},
        {"pos": [7, 8], "count": 35, "radius": 0},
        {"pos": [6, 1], "count": 30, "radius": 0},
        {"pos": [9, 5], "count": 28, "radius": 0},
    ],
    "hazards": [
        {"pos": [9, 0], "intensity": 0.88},
        {"pos": [9, 5], "intensity": 0.93},
        {"pos": [9, 9], "intensity": 0.88},
    ],
    "shelters": [
        {"id": "S1", "pos": [0, 2], "capacity": 100},
        {"id": "S2", "pos": [0, 7], "capacity": 100},
        {"id": "S3", "pos": [2, 5], "capacity": 80},
        {"id": "S4", "pos": [1, 0], "capacity": 60},
    ],
    "resources": {"drones": 5, "ambulances": 4},
    "score_weights": {
        "survival": 0.35,
        "evacuation_efficiency": 0.20,
        "shelter_balance": 0.15,
        "congestion": 0.15,
        "safety": 0.15,
    },
    "difficulty": "hard",
    "description": (
        "10x10 coastal city at hurricane landfall. Storm surge advances from south edge "
        "northward. 4 coastal roads pre-flooded, storm intensifies blocking more each step. "
        "238 civilians (many elderly) must reach inland shelters before surge cuts off routes."
    ),
}
