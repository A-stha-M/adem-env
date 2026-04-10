"""
ADEM OpenEnv Server — FastAPI application.
Runs via: uvicorn server:app --host 0.0.0.0 --port 8000
Or via:   uv run server
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adem.env import ADEMEnvironment
from adem.grader import ADEMGrader
from adem.models import ADEMAction, ADEMObservation
from adem.tasks import TASKS

app = FastAPI(
    title="ADEM — Adaptive Disaster Evacuation Management",
    description="OpenEnv-compliant RL environment for disaster evacuation AI research.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active environment instance (single-session server)
_env: Optional[ADEMEnvironment] = None


# ──────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: ADEMObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task: str = "controlled_evacuation"
    seed: Optional[int] = None


# ──────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "env": "adem", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {
                "name": name,
                "difficulty": ("easy" if name == "controlled_evacuation"
                               else "medium" if name == "dynamic_hazard"
                               else "hard"),
                "grid_size": cfg["grid_size"],
                "max_steps": cfg["max_steps"],
            }
            for name, cfg in TASKS.items()
        ]
    }


@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest):
    """Reset the environment for a given task. Returns first observation."""
    global _env
    if req.task not in TASKS:
        raise HTTPException(400, f"Unknown task '{req.task}'. Valid: {list(TASKS.keys())}")

    _env = ADEMEnvironment(task=req.task, seed=req.seed)
    obs = _env.reset()
    return StepResponse(observation=obs, reward=0.0, done=False, info={"message": "reset ok"})


@app.post("/step", response_model=StepResponse)
def step(action: ADEMAction):
    """Advance one timestep with the provided action."""
    global _env
    if _env is None:
        raise HTTPException(400, "Call /reset before /step")
    obs, reward, done, info = _env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    """Return current episode state summary."""
    global _env
    if _env is None:
        raise HTTPException(400, "No active environment. Call /reset first.")
    return _env.state()


@app.get("/score")
def score():
    """Compute final episode score (call after episode ends)."""
    global _env
    if _env is None:
        raise HTTPException(400, "No active environment. Call /reset first.")
    return ADEMGrader.grade(_env)


@app.get("/")
def root():
    return {
        "name": "ADEM OpenEnv",
        "description": "Adaptive Disaster Evacuation Management RL Environment",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/score"],
    }