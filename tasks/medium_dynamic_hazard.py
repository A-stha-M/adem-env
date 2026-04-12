"""Task config: dynamic_hazard (medium)."""

TASK = {
    "grid_size": 8,
    "max_steps": 20,
    "seed": 123,
    "hazard_spread_rate": 0.15,
    "wind_direction": None,
    "panic_enabled": False,
    "initial_road_blockages": [],
    "aftershock_probability": 0.0,
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
    "difficulty": "medium",
    "description": (
        "8x8 city, wildfire spreading uniformly from NW corner. ~130 civilians in "
        "multiple clusters. 3 shelters along south/east edges. Agent must continuously "
        "adapt routes as roads become blocked by expanding fire."
    ),
}
