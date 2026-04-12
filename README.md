---
title: Adem Env
emoji: 🚨
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# 🚨 ADEM — Adaptive Disaster Evacuation Management

> An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a **central disaster command system**, dynamically routing civilian populations to safety under spreading multi-hazard scenarios.

## Key Contributions

- **9 distinct disaster scenarios** spanning 3 difficulty tiers across 6 real disaster types
- **Wind-driven hazard spread** — agents must route civilians upwind/perpendicular to wind
- **Earthquake aftershocks** — stochastic road blocking forces mid-episode re-planning
- **Panic crowd dynamics** — civilians partially ignore instructions based on local fear level
- **Triple simultaneous hazard** (`multi_hazard_city`) — frontier-model challenge with 3 concurrent disaster sources
- **Dense multi-component reward** — 6 reward signals per step, never sparse
- **Deterministic grading** — seeded RNG, no LLM judges, fully reproducible
- **History-aware baseline agent** — `inference.py` uses multi-turn conversation history so decisions build on prior observations/actions

## Why This Environment Is Challenging

- Civilians only partially comply under panic (compliance scales inversely with panic level)
- Roads block dynamically mid-episode (aftershocks, storm surge, fire spread)
- Wind makes hazard spread asymmetric — the "obvious" route may be the deadly one
- In `industrial_chemical`, shelters are **downwind** of the hazard — agents must race the plume
- `multi_hazard_city` has 3 simultaneous threats requiring quadrant-independent routing
- Shelter capacity is limited — naive "move everyone to one shelter" fails

## Real-World Motivation

Large-scale evacuations are still managed with **static, pre-planned routes** that assume roads stay open, hazards evolve slowly, and people follow instructions. In reality: roads block dynamically, panic causes crowd movement to defy instructions, shelters overcrowd unevenly, and hazards spread non-linearly.

**ADEM fills a clear gap** — no standardized RL benchmark exists for adaptive evacuation under dynamic hazards, crowd panic, and multi-objective tradeoffs.

This also reflects a task real human responders perform in emergency operations centers: they continuously re-route people based on evolving hazards, road closures, congestion, and shelter capacity while minimizing casualties under time pressure.

In practice, this maps to real roles and workflows:
- Incident commanders prioritize neighborhoods for evacuation waves.
- Transportation/control teams open, close, and re-route road corridors.
- Shelter coordinators rebalance inflow as capacities change.
- Field teams adapt plans after aftershocks, flooding, and communication delays.

**Potential users:**
- Emergency management agencies testing AI decision support systems
- Smart city researchers benchmarking crowd management algorithms
- RL safety researchers studying multi-objective policy learning
- Disaster response policy simulation

---

## Grid Architecture

Each task simulates a square N×N city grid:

```
Grid orientation (6×6 example — controlled_evacuation):
 col→  0    1    2    3    4    5
row 0 [🔥] [ P] [ P] [  ] [ P] [  ]
row 1 [  ] [ P] [ P] [  ] [ P] [  ]
row 2 [  ] [ P] [ P] [ ■] [  ] [  ]
row 3 [  ] [  ] [  ] [  ] [  ] [  ]
row 4 [  ] [  ] [  ] [  ] [ P] [  ]
row 5 [  ] [  ] [S1] [  ] [  ] [S2]

🔥 = Hazard source   P = Civilians   S = Shelter   ■ = Blocked road
N=row-1  S=row+1  E=col+1  W=col-1
```

**Wind example** (`building_fire`, wind="E"):
```
 col→  0    1    2    3    4    5    6
row 3 [🔥→] [→ ] [→P] [→P] [→ ] [→ ] [  ]
          fire spreads EAST via wind pressure
          civilians must move EAST to outrun it → shelters at col 6
```

**Earthquake blockages** (`earthquake_response`):
```
 ■ = Road blocked by structural collapse (pre-placed at episode start)
 ? = May block during episode (aftershock probability = 18%/step)
Agent can use road_controls to clear blocked roads (dispatch crews).
```

---

## Tasks

| Task | Difficulty | Grid | Steps | Civilians | Hazards | Novel Mechanic |
|------|:---:|:---:|:---:|:---:|:---:|---|
| `controlled_evacuation` | Easy | 6×6 | 15 | ~48 | 1 static | Baseline routing |
| `flash_flood` | Easy | 6×6 | 15 | 64 | Rising water | South-biased spread |
| `building_fire` | Easy | 7×7 | 15 | 85 | 1 wind-driven | East wind acceleration |
| `dynamic_hazard` | Medium | 8×8 | 20 | ~130 | 1 spreading | Continuous re-routing |
| `earthquake_response` | Medium | 8×8 | 20 | 125 | 2 + aftershocks | Pre-blocked + dynamic roads |
| `industrial_chemical` | Medium | 9×9 | 20 | 165 | East-drifting plume | Race the plume eastward |
| `panic_evacuation` | Hard | 10×10 | 25 | ~230 | 2 simultaneous | Panic compliance model |
| `hurricane_coastal` | Hard | 10×10 | 25 | 238 | 3 + storm surge | Pre-flooded + intensifying |
| `multi_hazard_city` | Hard | 10×10 | 30 | ~315 | 3 simultaneous | Triple-threat frontier test |

### Task Descriptions

**Easy: Controlled Evacuation** — A single static fire in the NW corner. Roads clear, no panic. The canonical "learn to route" baseline. Grader rewards survival (45%) and efficiency (25%).

**Easy: Flash Flood** — Floodwater rises from the north edge and spreads south rapidly (wind bias). The 3 shelters are all in the south. Civilians in rows 0-2 are immediately threatened. Speed bonus: evacuating >60% before step 10 adds a grader bonus.

**Easy: Building Fire** — A single fire on the west side is blown east by wind (2.5× east spread). Shelters are at the east corners. Civilians must move east to safety — but so does the fire. The agent must keep civilians ahead of the fire front.

**Medium: Dynamic Hazard** — Classic wildfire that spreads radially from NW. Routes that are clear at step 1 may be blocked by step 8. Agent must continuously reconsider routing decisions.

**Medium: Earthquake Response** — 5 roads pre-blocked by structural collapse. An 18%/step aftershock probability blocks more roads dynamically. 125 civilians are panicked. Grader bonus for clearing blocked roads using `road_controls`.

**Medium: Industrial Chemical** — Toxic plume starts on the west edge and drifts EAST via wind. Shelters are also on the east side. Civilians must race the plume east — or take a north/south detour to stay out of the plume corridor. Panic complicates compliance.

**Hard: Panic Evacuation** — Two simultaneous hazards from opposite corners. Over 230 civilians with active panic (initial 5-30% panic, growing with congestion). Shelter capacity is insufficient for all civilians if one shelter overflows. Drone deployment is critical.

**Hard: Hurricane Coastal** — Storm surge advances northward from the coast (row 9). 4 coastal roads are pre-flooded. The storm intensifies each step, blocking more roads (12%/step). 238 civilians — many elderly — must move inland before surge cuts off routes. Grader bonus for clearing the coastal strip.

**Hard: Multi-Hazard City** — Three simultaneous disasters: wildfire (NW), flood (SE), chemical (NE). 315+ civilians across all quadrants. Central junction pre-blocked (6 roads). Ongoing instability (15%/step). Each population cluster must escape in a different direction. No single routing strategy works — the agent must reason about each quadrant independently.

## Adding New Tasks

To add a new task:

1. Create a new file in `tasks/` (for example, `tasks/medium_my_new_task.py`).
2. Define a top-level `TASK = {...}` dictionary using any existing task file as the template.
3. Optional: define `TASK_ID = "my_new_task"` if you do not want the ID inferred from filename.

Done. The loader in `tasks/__init__.py` auto-discovers any task module that exports a `TASK` dictionary.

Task ID inference rules:
- Filenames prefixed with `easy_`, `medium_`, or `hard_` have that prefix removed.
- Example: `hard_coastal_failure.py` is registered as `coastal_failure`.

No changes are required in the core environment engine, server endpoints, or inference runner for standard tasks that follow the same schema.

Grader note:
- The grader is mostly generic and reads shared metrics from environment state.
- Task-specific bonuses currently exist for a few named tasks in `graders/grader.py`; new tasks will still work with default grading, and you can add a custom bonus rule there if needed.

## Project Structure

```text
adem-env/
├── models.py                  # Pydantic models: ADEMAction, ADEMObservation, ADEMReward
├── openenv.yaml               # OpenEnv manifest (tasks, models, deployment metadata)
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Packaging metadata
├── Dockerfile                 # Container image for local run/HF Spaces
├── inference.py               # Baseline multi-task inference runner
├── adem_env.py                # Async client wrapper used by inference
├── README.md
│
├── env/                       # Core simulation engine
│   ├── __init__.py
│   └── environment.py         # step() / reset() / state() dynamics
│
├── tasks/                     # Task definitions (auto-discovered)
│   ├── __init__.py            # Discovery loader -> TASKS registry
│   ├── easy_*.py              # Easy-tier scenarios
│   ├── medium_*.py            # Medium-tier scenarios
│   └── hard_*.py              # Hard-tier scenarios
│
├── graders/                   # Scoring logic
│   ├── __init__.py
│   └── grader.py              # Final score + task bonus rules
│
└── server/
    └── app.py                 # FastAPI API: /health, /tasks, /reset, /step, /state, /score
```

---

## Action Space

All actions are sent as a single JSON object:

```json
{
  "zone_directions": {"4,3": "S", "2,1": "E", "0,5": "W"},
  "road_controls": {"3,2": true},
  "resource_allocations": {"S1": 5},
  "deploy_drone": "5,5"
}
```

| Field | Type | Description |
|---|---|---|
| `zone_directions` | dict[str→str] | `"row,col"` → `"N"/"S"/"E"/"W"/"STAY"`. Directs civilians in that cell. Compliance reduced by panic. |
| `road_controls` | dict[str→bool] | `"row,col"` → `true` (open) / `false` (close). Opens earthquake debris, redirects flow. |
| `resource_allocations` | dict[str→int] | Shelter ID → resource units to allocate there. |
| `deploy_drone` | str or null | `"row,col"` — deploys a drone. Reduces panic by -0.35 at target, -0.10 in 4 adjacent cells. |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `population_grid` | NxN int | Civilian count per cell |
| `hazard_grid` | NxN float [0,1] | Hazard intensity — >0.65 kills civilians each step |
| `road_blockages` | NxN int | 0=open, 1=blocked |
| `shelter_capacities` | dict | Max occupancy per shelter |
| `shelter_populations` | dict | Currently evacuated count per shelter |
| `congestion_levels` | NxN float | Traffic density — high congestion amplifies panic |
| `panic_levels` | NxN float | Compliance reduction factor — 1.0 = completely ignoring orders |
| `vulnerable_population_map` | NxN int | Elderly/disabled count per cell |
| `available_resources` | dict | Drones and ambulances remaining |
| `time_remaining` | int | Steps left in episode |
| `total_population` | int | Initial population (constant) |
| `evacuated` | int | Cumulative safely sheltered |
| `casualties` | int | Cumulative deaths |
| `grid_size` | int | N |

---

## Reward Function

### Per-Step (Dense — emitted every timestep)

```
reward =
  + 0.050 × people_evacuated_this_step      (primary positive signal)
  + 0.010 × shelter_balance_score           (encourages load balancing)
  + 0.002 × cumulative_shelter_ratio        (vulnerable pop proxy)
  − 0.080 × casualties_this_step            (hard penalty for deaths)
  − 0.030 × mean_congestion × 10            (bottleneck penalty)
  − 0.040 × high_risk_pop_ratio × 10        (people in hazard zones)
  − 0.002                                   (time pressure — act fast)
```

### Final Score (0.0 – 1.0, deterministic)

```
score = w_survival    × survival_rate           (0.35–0.50 by task)
      + w_efficiency  × evacuation_efficiency   (0.20–0.30 by task)
      + w_balance     × shelter_balance         (0.08–0.15 by task)
      + w_congestion  × congestion_score        (0.08–0.20 by task)
      + w_safety      × safety_score            (0.07–0.15 by task)
      + task_bonus                              (up to 0.045, task-specific)
```

**Sub-metrics:**
| Metric | Formula | Range |
|---|---|---|
| `survival_rate` | (initial_pop − casualties) / initial_pop | [0, 1] |
| `evacuation_efficiency` | evacuated/total × (1 − 0.5×steps_used/max_steps) | [0, 1] |
| `shelter_balance` | 1 − 5×variance(utilisation_ratios) | [0, 1] |
| `congestion_score` | 1 − mean(congestion_grid) | [0, 1] |
| `safety_score` | 1 − high_risk_pop/total | [0, 1] |

**Task bonuses** (task-specific excellence rewards):
- `flash_flood`: +0.04 if >60% evacuated before step 10
- `earthquake_response`: up to +0.03 for clearing pre-blocked roads
- `hurricane_coastal`: +0.04 for clearing all civilians from coastal strip
- `multi_hazard_city`: +0.015 per hazard origin zone cleared of population

### Per-Step Shaping (Includes Penalties)

Negative per-step rewards are expected in ADEM when casualties, congestion, or hazard exposure rises.

| Signal | Reward Contribution |
|---|---:|
| Evacuating civilians | `+0.050 × people_evacuated_this_step` |
| Casualties | `-0.080 × casualties_this_step` |
| Congestion pressure | `-0.030 × mean_congestion × 10` |
| Population in high-risk cells | `-0.040 × high_risk_pop_ratio × 10` |
| Time pressure (every step) | `-0.002` |

Because of this shaping, step-level rewards can be negative even when the policy is partially correct. Final graded score remains normalized to `[0, 1]`.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "env": "adem", "version": "1.0.0"}` |
| `/tasks` | GET | List all 9 tasks with metadata |
| `/reset` | POST | Start new episode. Body: `{"task": "...", "seed": null}` |
| `/step` | POST | Advance one timestep. Body: `ADEMAction` JSON |
| `/state` | GET | Current episode state summary |
| `/score` | GET | Final episode score (call after episode ends) |

### API Usage

```bash
# List available tasks
curl http://localhost:7860/tasks

# Reset (start a new episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "controlled_evacuation", "seed": null}'

# Step (take an action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "zone_directions": {"1,1": "S", "2,3": "E"},
    "road_controls": {"3,2": true},
    "resource_allocations": {},
    "deploy_drone": "4,4"
  }'

# Get current episode state
curl http://localhost:7860/state

# Get final score
curl http://localhost:7860/score
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/adem-env
cd adem-env
pip install -r requirements.txt
```

## Local Testing

```bash
# Terminal 1: Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2: Run inference
set HF_TOKEN=hf_your_token_here
set ADEM_SERVER_URL=http://localhost:7860
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Running Inference

```bash
export API_KEY=your_api_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export ADEM_SERVER_URL=https://astha28-adem-env.hf.space

python inference.py
```

The inference script runs a baseline LLM agent against configured tasks and emits structured logs:

```text
[START] task=controlled_evacuation env=adem model=llama-3.3-70b-versatile
[STEP] step=1 action={"zone_directions":{"1,1":"S"}} reward=-0.01 done=false error=null
[STEP] step=2 action={"zone_directions":{"2,3":"E"}} reward=0.24 done=false error=null
[STEP] step=3 action={"zone_directions":{"3,4":"E"}} reward=0.09 done=false error=null
[END] success=true steps=12 score=0.811 rewards=-0.09,-0.01,...
```

### Baseline Inference Design

- Uses the OpenAI Python client (`from openai import OpenAI`) against OpenAI-compatible endpoints.
- Maintains multi-turn history (`MAX_HISTORY_TURNS`) so each action is conditioned on recent trajectory context.
- Emits reproducible structured logs (`[START]`, `[STEP]`, `[END]`) for task-level scoring and auditability.
- Supports strict benchmark mode (`STRICT_BASELINE=1`) to fail fast on auth/model/parse errors and prevent silent heuristic-only scores.

## Docker

```bash
docker build -t adem-env:latest .
docker run -d -p 7860:7860 --name adem adem-env:latest

# Wait ~10s, then test
curl http://localhost:7860/health

# Run inference against container
set ADEM_SERVER_URL=http://localhost:7860
python inference.py

docker stop adem && docker rm adem
```

---

## Baseline Scores

All baselines below were run via Groq free tier.

| Task | Difficulty | Llama 3.1 8B | Llama 4 Scout 17B | Llama 3.3 70B |
|------|:---:|:---:|:---:|:---:|
| `controlled_evacuation` | Easy | 0.823 | 0.762 | 0.811 |
| `flash_flood` | Easy | 0.389 | 0.395 | 0.461 |
| `building_fire` | Easy | 0.579 | 0.609 | 0.712 |
| `dynamic_hazard` | Medium | 0.745 | 0.734 | 0.740 |
| `earthquake_response` | Medium | 0.000 | 0.000 | 0.770 |
| `industrial_chemical` | Medium | 0.675 | 0.680 | 0.692 |
| `panic_evacuation` | Hard | 0.731 | 0.775 | 0.778 |
| `hurricane_coastal` | Hard | 0.000 | 0.000 | 0.504 |
| `multi_hazard_city` | Hard | 0.000 | 0.685 | 0.740 |
| **Average** | | **0.438** | **0.516** | **0.690** |

Llama 3.3 70B values were assembled across resumed runs due intermittent Groq free-tier HTTP 429 throttling.

Key observations:
- No single model dominates all tasks: the 8B model is stable on several tasks, while Scout 17B peaks on `panic_evacuation` and `multi_hazard_city`.
- `flash_flood` is deceptively difficult for all models because early water spread quickly collapses viable corridors.
- `earthquake_response` and `hurricane_coastal` remain high-variance failure points under strict JSON/action requirements.
- Larger models show stronger multi-region reasoning in some hard tasks, but rate limits can erase that advantage in long runs.
- Scores are non-deterministic due LLM sampling and parser sensitivity; moderate variance across reruns is expected.

### Trusted Baseline Protocol 

Run with strict mode so authentication/model errors cannot produce fallback-based scores:

```bat
set ADEM_SERVER_URL=https://astha28-adem-env.hf.space
set API_BASE_URL=https://api.groq.com/openai/v1
set API_KEY=<your_groq_api_key>
set STRICT_BASELINE=1
set ALLOW_HEURISTIC_FALLBACK=0
set PRECHECK_MODEL_ACCESS=1
```

Then run per model:

```bat
set MODEL_NAME=llama-3.1-8b-instant
python inference.py 2>&1 | findstr "[END]"

set MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
python inference.py 2>&1 | findstr "[END]"

set MODEL_NAME=llama-3.3-70b-versatile
python inference.py 2>&1 | findstr "[END]"
```

Accept runs only if:
- No `llm_error:` appears in `[STEP]` lines.
- No auth/model-access failures appear in `[DEBUG]` lines.
- Each model yields a complete 9-task `[END]` set.

---

## Environment Variables

Authentication note:
- Set only one of `OPENAI_API_KEY`, `HF_TOKEN`, or `API_KEY`.
- The script resolves keys in this order: `OPENAI_API_KEY` -> `HF_TOKEN` -> `API_KEY`.

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI-compatible API key (preferred) | Not set |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace token (fallback if `OPENAI_API_KEY` is unset) | Not set |
| `API_KEY` | Generic API key (final fallback) | Not set |
| `ADEM_TASKS` | Comma-separated task IDs to run in inference | `controlled_evacuation,flash_flood,building_fire,dynamic_hazard,earthquake_response,industrial_chemical,panic_evacuation,hurricane_coastal,multi_hazard_city` |
| `STRICT_BASELINE` | Fail fast on LLM API/parse errors | `1` |
| `ALLOW_HEURISTIC_FALLBACK` | Allow heuristic fallback when strict mode is off | `0` when strict mode is on |
| `PRECHECK_MODEL_ACCESS` | Run model-auth/access precheck before tasks | `1` when strict mode is on |
| `ADEM_SERVER_URL` | Connect to existing server (overrides Docker) | `https://astha28-adem-env.hf.space` |
| `ADEM_PORT` | Server port | `7860` |

---

## OpenEnv Spec Compliance

- ✅ `openenv.yaml` with full metadata, 9 task definitions, typed model descriptions
- ✅ `step(action)` → `(observation, reward, done, info)`
- ✅ `reset()` → initial observation (clean state, seeded RNG)
- ✅ `state()` → current episode metadata
- ✅ Typed Pydantic models: `ADEMObservation`, `ADEMAction`, `ADEMReward`
- ✅ 9 tasks across 3 difficulty tiers with programmatic graders
- ✅ Scores in [0.0, 1.0] with dense partial progress signals
- ✅ Fully deterministic given same seed
- ✅ Working Dockerfile for containerized execution
- ✅ Baseline `inference.py` with mandatory `[START]`/`[STEP]`/`[END]` log format

