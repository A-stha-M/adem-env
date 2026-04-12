"""Task config: controlled_evacuation (easy)."""

TASK = {
    "grid_size": 6,
    "max_steps": 15,
    "seed": 42,
    "hazard_spread_rate": 0.0,
    "wind_direction": None,
    "panic_enabled": False,
    "initial_road_blockages": [],
    "aftershock_probability": 0.0,
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
    "difficulty": "easy",
    "description": (
        "6x6 urban grid, single static fire in the NW corner, ~48 civilians, "
        "2 shelters along the south edge. Teaches basic directional routing."
    ),
}
