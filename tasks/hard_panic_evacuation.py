"""Task config: panic_evacuation (hard)."""

TASK = {
    "grid_size": 10,
    "max_steps": 25,
    "seed": 777,
    "hazard_spread_rate": 0.20,
    "wind_direction": None,
    "panic_enabled": True,
    "initial_road_blockages": [],
    "aftershock_probability": 0.0,
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
    "difficulty": "hard",
    "description": (
        "10x10 city, 2 simultaneous hazards (NW fire + SE flood), 230+ civilians with "
        "active panic dynamics (compliance ~35%). 4 shelters with limited capacity. "
        "Agent must manage crowd psychology while distributing population across shelters."
    ),
}
