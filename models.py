"""
Typed Pydantic models for the ADEM OpenEnv environment.
Observation, Action, and Reward follow the OpenEnv spec.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ADEMObservation(BaseModel):
    """Full observable state of the disaster evacuation environment."""
    population_grid: List[List[int]] = Field(
        description="NxN grid - number of civilians in each cell"
    )
    hazard_grid: List[List[float]] = Field(
        description="NxN grid - hazard intensity per cell [0.0, 1.0]"
    )
    road_blockages: List[List[int]] = Field(
        description="NxN grid - 0=passable, 1=blocked"
    )
    shelter_capacities: Dict[str, int] = Field(
        description="shelter_id -> maximum occupancy"
    )
    shelter_populations: Dict[str, int] = Field(
        description="shelter_id -> current occupants (safely evacuated)"
    )
    congestion_levels: List[List[float]] = Field(
        description="NxN grid - traffic congestion [0.0, 1.0]"
    )
    panic_levels: List[List[float]] = Field(
        description="NxN grid - population panic level [0.0, 1.0]"
    )
    vulnerable_population_map: List[List[int]] = Field(
        description="NxN grid - count of mobility-impaired / elderly per cell"
    )
    available_resources: Dict[str, int] = Field(
        description="resource_type -> count (drones, ambulances)"
    )
    time_remaining: int = Field(description="Steps left in episode")
    total_population: int = Field(description="Initial total population at episode start")
    evacuated: int = Field(description="People safely in shelters so far")
    casualties: int = Field(description="Deaths caused by hazard exposure")
    grid_size: int = Field(description="Side length N of the square grid")


class ADEMAction(BaseModel):
    """
    Agent's action for one timestep.
    All fields are optional - omitted fields = no change to that control.
    """
    zone_directions: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map 'row,col' -> direction. Direction in {N, S, E, W, STAY}. "
            "Tells civilians in that cell which way to move."
        )
    )
    road_controls: Dict[str, bool] = Field(
        default_factory=dict,
        description="Map 'row,col' -> True (open) / False (close). Override road state."
    )
    resource_allocations: Dict[str, int] = Field(
        default_factory=dict,
        description="Map shelter_id -> additional resource units to send there."
    )
    deploy_drone: Optional[str] = Field(
        default=None,
        description="Zone 'row,col' to deploy a drone - reduces panic in that zone."
    )


class ADEMReward(BaseModel):
    """Decomposed reward for transparency and debugging."""
    total: float
    evacuation_component: float
    vulnerable_saved_component: float
    shelter_balance_component: float
    casualty_penalty: float
    congestion_penalty: float
    hazard_exposure_penalty: float
