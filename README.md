---
title: Adem Env
emoji: 👀
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
---

# 🚨 ADEM — Adaptive Disaster Evacuation Management

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a central disaster command system, dynamically routing civilians to safety during spreading hazards.

## 🎯 Real-World Motivation

Large-scale evacuations during disasters (floods, wildfires, earthquakes) are still managed with static plans. This environment benchmarks adaptive AI decision-making under:
- **Dynamic hazard spread** (fire/flood blocking roads in real time)
- **Panic dynamics** (population partially ignores instructions)
- **Multi-objective tradeoffs** (speed vs safety vs shelter balance)
- **Limited resources** (drones, ambulances, shelter capacity)

## 🗺️ Environment Description

A discrete grid-based city simulation. The agent is the central command system issuing per-timestep evacuation orders.

### Action Space
```json
{
  "zone_directions": {"row,col": "N|S|E|W|STAY"},
  "road_controls":   {"row,col": true},
  "resource_allocations": {"shelter_id": 5},
  "deploy_drone": "row,col"
}
```

### Observation Space
| Field | Type | Description |
|---|---|---|
| `population_grid` | NxN int | People per cell |
| `hazard_grid` | NxN float [0,1] | Hazard intensity |
| `road_blockages` | NxN int | 0=open, 1=blocked |
| `shelter_capacities` | dict | Max occupancy per shelter |
| `shelter_populations` | dict | Current safe evacuees |
| `congestion_levels` | NxN float | Traffic density |
| `panic_levels` | NxN float | Compliance reduction factor |
| `time_remaining` | int | Steps left |
| `evacuated` | int | Total safely sheltered |
| `casualties` | int | Deaths from hazard |

## 📋 Tasks

| Task | Difficulty | Grid | Max Steps | Description |
|---|---|---|---|---|
| `controlled_evacuation` | Easy | 6×6 | 15 | Static hazard, 2 shelters, ~50 civilians |
| `dynamic_hazard` | Medium | 8×8 | 20 | Spreading fire, 3 shelters, ~130 civilians |
| `panic_evacuation` | Hard | 10×10 | 25 | Dual hazards, panic dynamics, 4 shelters, ~230 civilians |

## 📊 Reward Function

**Per-step (dense):**
```
reward = +0.05 × evacuated_this_step
       + 0.02 × vulnerable_saved_ratio
       + 0.01 × shelter_balance
       - 0.08 × casualties_this_step
       - 0.03 × mean_congestion
       - 0.04 × high_risk_population_ratio
```

**Final score (0.0–1.0):**
```
score = 0.35 × survival_rate
      + 0.20 × evacuation_efficiency
      + 0.15 × shelter_balance
      + 0.15 × congestion_score
      + 0.15 × safety_score
```

## 🚀 Setup

```bash
git clone https://github.com/YOUR_USERNAME/adem-env
cd adem-env
pip install -r requirements.txt
```

## 🧪 Local Testing

```bash
# Start server
uvicorn server:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
export HF_TOKEN=hf_your_token
export ADEM_SERVER_URL=http://localhost:8000
python inference.py
```

## 🐳 Docker

```bash
docker build -t adem-env .
docker run -p 8000:8000 adem-env

# Run inference against Docker container
export ADEM_SERVER_URL=http://localhost:8000
python inference.py
```

## 📈 Baseline Scores

| Task | Model | Score |
|---|---|---|
| `controlled_evacuation` | Qwen2.5-72B | ~0.55 |
| `dynamic_hazard` | Qwen2.5-72B | ~0.38 |
| `panic_evacuation` | Qwen2.5-72B | ~0.22 |

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks |
| `/reset` | POST | Start new episode |
| `/step` | POST | Advance one timestep |
| `/state` | GET | Current episode state |
| `/score` | GET | Final episode score |