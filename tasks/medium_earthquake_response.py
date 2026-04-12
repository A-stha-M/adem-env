"""Task config: earthquake_response (medium)."""

TASK = {
    "grid_size": 8,
    "max_steps": 20,
    "seed": 314,
    "hazard_spread_rate": 0.06,
    "wind_direction": None,
    "panic_enabled": True,
    "initial_road_blockages": [
        [1, 3], [2, 5], [4, 2], [5, 6], [3, 0],
    ],
    "aftershock_probability": 0.18,
    "population_centers": [
        {"pos": [2, 2], "count": 35, "radius": 0},
        {"pos": [5, 5], "count": 35, "radius": 0},
        {"pos": [1, 6], "count": 20, "radius": 0},
        {"pos": [6, 1], "count": 20, "radius": 0},
        {"pos": [4, 4], "count": 15, "radius": 0},
    ],
    "hazards": [
        {"pos": [3, 3], "intensity": 0.72},
        {"pos": [0, 7], "intensity": 0.65},
    ],
    "shelters": [
        {"id": "S1", "pos": [7, 0], "capacity": 65},
        {"id": "S2", "pos": [7, 7], "capacity": 65},
        {"id": "S3", "pos": [0, 3], "capacity": 40},
    ],
    "resources": {"drones": 3, "ambulances": 3},
    "score_weights": {
        "survival": 0.35,
        "evacuation_efficiency": 0.20,
        "shelter_balance": 0.15,
        "congestion": 0.15,
        "safety": 0.15,
    },
    "difficulty": "medium",
    "description": (
        "8x8 city post-earthquake. 5 roads pre-blocked by structural collapse. "
        "Aftershocks randomly block additional roads each step. 125 panicked civilians. "
        "Agent must assess network topology and re-route dynamically around new blockages."
    ),
}
