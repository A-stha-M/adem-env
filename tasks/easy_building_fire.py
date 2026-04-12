"""Task config: building_fire (easy)."""

TASK = {
    "grid_size": 7,
    "max_steps": 15,
    "seed": 256,
    "hazard_spread_rate": 0.12,
    "wind_direction": "E",
    "panic_enabled": False,
    "initial_road_blockages": [],
    "aftershock_probability": 0.0,
    "population_centers": [
        {"pos": [3, 2], "count": 25, "radius": 0},
        {"pos": [2, 4], "count": 20, "radius": 0},
        {"pos": [4, 3], "count": 18, "radius": 0},
        {"pos": [1, 5], "count": 12, "radius": 0},
        {"pos": [5, 5], "count": 10, "radius": 0},
    ],
    "hazards": [
        {"pos": [3, 0], "intensity": 0.92},
    ],
    "shelters": [
        {"id": "S1", "pos": [0, 6], "capacity": 50},
        {"id": "S2", "pos": [6, 6], "capacity": 50},
    ],
    "resources": {"drones": 1, "ambulances": 2},
    "score_weights": {
        "survival": 0.40,
        "evacuation_efficiency": 0.30,
        "shelter_balance": 0.10,
        "congestion": 0.10,
        "safety": 0.10,
    },
    "difficulty": "easy",
    "description": (
        "7x7 urban block. A structural fire starts on the west side and is blown EAST "
        "by wind. 85 civilians must evacuate east toward corner shelters while "
        "outrunning the wind-driven fire spread."
    ),
}
