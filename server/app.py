"""
ADEM OpenEnv Server — FastAPI application.
Runs via: uvicorn server.app:app --host 0.0.0.0 --port 8000
Or via:   uv run server
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import ADEMEnvironment
from graders import ADEMGrader
from models import ADEMAction, ADEMObservation
from tasks import TASKS

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

_env: Optional[ADEMEnvironment] = None


class StepResponse(BaseModel):
    observation: ADEMObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task: str = "controlled_evacuation"
    seed: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok", "env": "adem", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": name,
                "difficulty": (
                    "easy" if name == "controlled_evacuation"
                    else "medium" if name == "dynamic_hazard"
                    else "hard"
                ),
                "grid_size": cfg["grid_size"],
                "max_steps": cfg["max_steps"],
            }
            for name, cfg in TASKS.items()
        ]
    }


@app.post("/reset", response_model=StepResponse)
def reset(req: Optional[ResetRequest] = None):
    global _env
    req = req or ResetRequest()
    if req.task not in TASKS:
        raise HTTPException(400, f"Unknown task '{req.task}'")

    _env = ADEMEnvironment(task=req.task, seed=req.seed)
    obs = _env.reset()
    return StepResponse(
        observation=obs,
        reward=0.0,
        done=False,
        info={"message": "reset ok"},
    )


@app.post("/step", response_model=StepResponse)
def step(action: ADEMAction):
    global _env
    if _env is None:
        raise HTTPException(400, "Call /reset before /step")

    obs, reward, done, info = _env.step(action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(400, "No active environment")
    return _env.state()


@app.get("/score")
def score():
    global _env
    if _env is None:
        raise HTTPException(400, "No active environment")
    return ADEMGrader.grade(_env)


@app.get("/")
def root():
    return {
        "name": "ADEM OpenEnv",
        "description": "Adaptive Disaster Evacuation Management RL Environment",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/score"],
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()