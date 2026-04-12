"""Task config: industrial_chemical (medium)."""

TASK = {
    "grid_size": 9,
    "max_steps": 20,
    "seed": 512,
    "hazard_spread_rate": 0.18,
    "wind_direction": "E",
    "panic_enabled": True,
    "initial_road_blockages": [],
    "aftershock_probability": 0.0,
    "population_centers": [
        {"pos": [2, 2], "count": 40, "radius": 0},
        {"pos": [5, 3], "count": 35, "radius": 0},
        {"pos": [7, 5], "count": 30, "radius": 0},
        {"pos": [3, 7], "count": 25, "radius": 0},
        {"pos": [1, 5], "count": 20, "radius": 0},
        {"pos": [6, 7], "count": 15, "radius": 0},
    ],
    "hazards": [
        {"pos": [4, 0], "intensity": 0.95},
        {"pos": [5, 0], "intensity": 0.88},
    ],
    "shelters": [
        {"id": "S1", "pos": [0, 8], "capacity": 70},
        {"id": "S2", "pos": [8, 8], "capacity": 70},
        {"id": "S3", "pos": [4, 8], "capacity": 55},
    ],
    "resources": {"drones": 3, "ambulances": 2},
    "score_weights": {
        "survival": 0.35,
        "evacuation_efficiency": 0.20,
        "shelter_balance": 0.15,
        "congestion": 0.15,
        "safety": 0.15,
    },
    "difficulty": "medium",
    "description": (
        "9x9 industrial district. Chemical plant leak on west edge. Toxic plume drifts "
        "EAST via wind - shelters are also on the EAST side. 165 civilians must race the "
        "plume eastward. North/south detour may be safer but slower: key routing tradeoff."
    ),
}
