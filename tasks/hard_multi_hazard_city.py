"""Task config: multi_hazard_city (hard)."""

TASK = {
    "grid_size": 10,
    "max_steps": 30,
    "seed": 1337,
    "hazard_spread_rate": 0.20,
    "wind_direction": None,
    "panic_enabled": True,
    "initial_road_blockages": [
        [4, 4], [4, 5], [5, 4], [5, 5],
        [2, 7], [7, 2],
    ],
    "aftershock_probability": 0.15,
    "population_centers": [
        {"pos": [2, 2], "count": 60, "radius": 1},
        {"pos": [2, 7], "count": 55, "radius": 1},
        {"pos": [7, 2], "count": 55, "radius": 1},
        {"pos": [7, 7], "count": 50, "radius": 1},
        {"pos": [4, 4], "count": 40, "radius": 0},
        {"pos": [1, 5], "count": 30, "radius": 0},
        {"pos": [5, 1], "count": 25, "radius": 0},
    ],
    "hazards": [
        {"pos": [0, 0], "intensity": 0.90},
        {"pos": [9, 9], "intensity": 0.85},
        {"pos": [0, 9], "intensity": 0.82},
    ],
    "shelters": [
        {"id": "S1", "pos": [9, 0], "capacity": 110},
        {"id": "S2", "pos": [0, 5], "capacity": 90},
        {"id": "S3", "pos": [5, 9], "capacity": 90},
        {"id": "S4", "pos": [9, 5], "capacity": 80},
        {"id": "S5", "pos": [5, 0], "capacity": 80},
    ],
    "resources": {"drones": 6, "ambulances": 5},
    "score_weights": {
        "survival": 0.35,
        "evacuation_efficiency": 0.20,
        "shelter_balance": 0.15,
        "congestion": 0.15,
        "safety": 0.15,
    },
    "difficulty": "hard",
    "description": (
        "10x10 mega-city, 3 SIMULTANEOUS hazards: wildfire NW + flood SE + chemical NE. "
        "315+ civilians across all quadrants. Central district pre-blocked (6 roads). "
        "Ongoing instability. 5 directional shelters - each population cluster must escape "
        "in a different direction. Frontier-model challenge requiring multi-threat reasoning."
    ),
}
