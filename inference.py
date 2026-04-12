"""
ADEM Inference Script
=====================
Runs the LLM agent against all 9 ADEM tasks and emits structured stdout logs.

Uses multi-turn conversation history so the LLM can see its own past actions
and their outcomes — significantly improving sequential decision-making.

Environment variables:
    API_BASE_URL        LLM endpoint (default: HuggingFace router)
    MODEL_NAME          LLM model identifier
    HF_TOKEN            HuggingFace / API key
    LOCAL_IMAGE_NAME    Docker image name (optional)
    ADEM_SERVER_URL     Connect to existing server URL (overrides docker startup)

Stdout format (mandatory):
    [START] task=<name> env=adem model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<str|null>
    [END]   success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from adem_env import ADEMAction, ADEMEnv, ADEMObservation

# ── Configuration ──────────────────────────────────────────────────────────────
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")
ADEM_SERVER_URL: Optional[str] = os.getenv("ADEM_SERVER_URL")

API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK: str = "adem"

TASKS_TO_RUN: List[str] = [
    "controlled_evacuation",   # Easy
    "flash_flood",             # Easy
    "building_fire",           # Easy
    "dynamic_hazard",          # Medium
    "earthquake_response",     # Medium
    "industrial_chemical",     # Medium
    "panic_evacuation",        # Hard
    "hurricane_coastal",       # Hard
    "multi_hazard_city",       # Hard
]

MAX_STEPS: int = 12           # Hard cap per episode (within task max_steps)
TEMPERATURE: float = 0.15     # Low temperature for more deterministic routing
MAX_TOKENS: int = 600
SUCCESS_SCORE_THRESHOLD: float = 0.30
MAX_HISTORY_TURNS: int = 4    # Multi-turn: keep last N (obs, action) pairs in context

# ── Mandatory Logging Format ───────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action[:120].replace("\n", " ") if action else "null"
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} "
        f"done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation Formatting ─────────────────────────────────────────────────────

def summarize_observation(obs: ADEMObservation, step: int, task: str) -> str:
    """Convert grid observation to a concise natural-language summary for the LLM."""
    n = obs.grid_size

    # Occupied zones (limit to top 15 to keep prompt concise)
    pop_zones = []
    for i in range(n):
        for j in range(n):
            p = obs.population_grid[i][j]
            if p > 0:
                hz = obs.hazard_grid[i][j]
                panic = obs.panic_levels[i][j]
                tag = ""
                if hz > 0.65:
                    tag = " ⚠️DANGER"
                elif hz > 0.40:
                    tag = " ⚡RISK"
                pop_zones.append(
                    f"  ({i},{j}): {p} people | hazard={hz:.2f} | panic={panic:.2f}{tag}"
                )
    pop_zones = pop_zones[:15]  # cap for prompt length

    # Active hazard zones
    hazard_zones = [
        f"  ({i},{j})={obs.hazard_grid[i][j]:.2f}"
        for i in range(n) for j in range(n)
        if obs.hazard_grid[i][j] > 0.40
    ][:12]

    # Blocked roads
    blocked = [
        f"({i},{j})"
        for i in range(n) for j in range(n)
        if obs.road_blockages[i][j] == 1
    ][:10]

    # Shelter status
    shelters_info = [
        f"  {sid}: {obs.shelter_populations[sid]}/{obs.shelter_capacities[sid]} filled"
        for sid in obs.shelter_capacities
    ]

    lines = [
        f"━━━ Step {step} | Task: {task} | Grid {n}×{n} | Time left: {obs.time_remaining} ━━━",
        f"Population: total={obs.total_population} | evacuated={obs.evacuated} "
        f"| casualties={obs.casualties} | in-city={obs.total_population - obs.evacuated - obs.casualties}",
        "",
        "CIVILIANS (cells with people):",
        *(pop_zones if pop_zones else ["  (none remaining)"]),
        "",
        f"HAZARD ZONES (>{0.4:.1f} intensity): " + (", ".join(hazard_zones) if hazard_zones else "none"),
        f"BLOCKED ROADS: " + (", ".join(blocked) if blocked else "none"),
        "",
        "SHELTERS:",
        *shelters_info,
        f"RESOURCES: {obs.available_resources}",
    ]
    return "\n".join(lines)


def get_shelter_hint(task: str) -> str:
    """Return shelter locations as a string hint for the LLM."""
    from tasks import TASKS
    cfg = TASKS.get(task, {})
    shelters = cfg.get("shelters", [])
    parts = [
        f"{s['id']}→({s['pos'][0]},{s['pos'][1]}) cap={s['capacity']}"
        for s in shelters
    ]
    return " | ".join(parts)


def get_task_hint(task: str) -> str:
    """Return a short strategic hint for the specific task."""
    from tasks import TASKS
    cfg = TASKS.get(task, {})
    wind = cfg.get("wind_direction")
    panic = cfg.get("panic_enabled", False)
    aftershock = cfg.get("aftershock_probability", 0.0)
    initial_blocks = len(cfg.get("initial_road_blockages", []))
    hazard_count = len(cfg.get("hazards", []))

    hints = []
    if wind:
        hints.append(f"Wind blows {wind} — hazard spreads fastest in {wind} direction")
    if panic:
        hints.append("PANIC MODE: civilians only ~35% compliant — use drones to calm zones")
    if aftershock > 0:
        hints.append(f"AFTERSHOCKS: {int(aftershock*100)}% chance roads block each step — check blockages")
    if initial_blocks > 0:
        hints.append(f"{initial_blocks} roads pre-blocked — route around them via road_controls")
    if hazard_count > 1:
        hints.append(f"{hazard_count} simultaneous hazards — each quadrant needs separate routing")

    return " | ".join(hints) if hints else "Standard evacuation — route all civilians to nearest shelter"


# ── LLM System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI disaster evacuation commander. Maximize civilian survival.

ENVIRONMENT:
- NxN grid. Cells = "row,col" (0-indexed, row 0 = top, col 0 = left).
- Hazard (fire/flood/chemical) spreads and kills civilians (intensity >0.65 = lethal).
- Shelters: civilians moving INTO a shelter cell are safely evacuated.
- Roads: blocked cells cannot be entered. You can open them with road_controls.
- Panic: reduces civilian compliance. Deploy drones to reduce panic.
- Wind: biases hazard spread in one direction (check task hint).

ACTION FORMAT — respond with ONLY valid JSON, no markdown, no explanation:
{
  "zone_directions": {"row,col": "DIRECTION", ...},
  "road_controls": {"row,col": true},
  "resource_allocations": {},
  "deploy_drone": "row,col"
}

DIRECTIONS: N=up(row-1), S=down(row+1), E=right(col+1), W=left(col-1), STAY

STRATEGY RULES:
1. Issue directions for EVERY cell with civilians — idle civilians die.
2. Route civilians toward nearest shelter, avoiding hazard and blocked roads.
3. If a road is blocked, use road_controls to open it: {"row,col": true}
4. In panic mode, deploy_drone on the most congested zone first.
5. Balance shelter loads — spread civilians across multiple shelters.
6. Move coastal/high-hazard populations FIRST — prioritize highest risk.
7. Check wind direction: route civilians PERPENDICULAR to wind if possible.

Respond with ONLY the JSON object. No text before or after.
""").strip()


# ── LLM Call with Multi-Turn History ──────────────────────────────────────────

def call_llm(
    client: OpenAI,
    obs: ADEMObservation,
    step: int,
    task: str,
    conversation_history: List[Dict[str, str]],
) -> ADEMAction:
    """
    Call LLM with multi-turn conversation history.
    The LLM sees its past observations AND past actions, enabling trajectory-aware planning.
    Falls back to heuristic on parse failure.
    """
    shelter_hint = get_shelter_hint(task)
    task_hint = get_task_hint(task)
    current_obs = summarize_observation(obs, step, task)

    user_content = textwrap.dedent(f"""
        {current_obs}

        SHELTER LOCATIONS: {shelter_hint}
        TASK HINT: {task_hint}

        Issue evacuation orders now. Route ALL civilians toward shelters.
        Respond with ONLY valid JSON.
    """).strip()

    # Build multi-turn message list
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include last MAX_HISTORY_TURNS (obs, action) pairs for trajectory awareness
    recent_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]
    messages.extend(recent_history)

    # Current observation
    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        raw = raw.strip()
        # Find JSON object boundaries
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        parsed = json.loads(raw)

        # Sanitise: remove any direction values the env doesn't recognise
        valid_dirs = {"N", "S", "E", "W", "STAY"}
        zone_dirs = {
            k: v.upper().strip()
            for k, v in parsed.get("zone_directions", {}).items()
            if v.upper().strip() in valid_dirs
        }
        parsed["zone_directions"] = zone_dirs
        return ADEMAction(**parsed)

    except Exception as exc:
        print(f"[DEBUG] LLM parse error step={step}: {exc}", flush=True)
        return _heuristic_action(obs, task)


def _heuristic_action(obs: ADEMObservation, task: str) -> ADEMAction:
    """
    Fallback heuristic: route every populated zone toward its nearest shelter.
    Uses Manhattan-distance nearest shelter computation.
    """
    from tasks import TASKS
    cfg = TASKS.get(task, {})
    shelters = cfg.get("shelters", [])
    n = obs.grid_size
    directions: Dict[str, str] = {}
    road_controls: Dict[str, bool] = {}

    # Try to open blocked roads near populated zones
    for i in range(n):
        for j in range(n):
            if obs.road_blockages[i][j] == 1:
                # Open if adjacent to population
                has_pop_neighbor = any(
                    0 <= i + di < n and 0 <= j + dj < n and obs.population_grid[i + di][j + dj] > 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if has_pop_neighbor:
                    road_controls[f"{i},{j}"] = True

    for i in range(n):
        for j in range(n):
            if obs.population_grid[i][j] <= 0:
                continue

            # Move away from high hazard first
            if obs.hazard_grid[i][j] > 0.65:
                # Emergency: move away from hazard
                best_dir = "S"
                if i == n - 1:
                    best_dir = "N"
                elif j < n // 2:
                    best_dir = "E"
                directions[f"{i},{j}"] = best_dir
                continue

            # Find nearest shelter by Manhattan distance
            if not shelters:
                directions[f"{i},{j}"] = "S" if i < n // 2 else "E"
                continue

            best_shelter = min(
                shelters,
                key=lambda s: abs(s["pos"][0] - i) + abs(s["pos"][1] - j),
            )
            sr, sc = best_shelter["pos"][0], best_shelter["pos"][1]

            dr = sr - i
            dc = sc - j

            # Move in direction of greatest distance component
            if abs(dr) >= abs(dc):
                d = "S" if dr > 0 else "N"
            else:
                d = "E" if dc > 0 else "W"

            # Check if that direction is blocked
            delta = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
            di, dj = delta[d]
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and obs.road_blockages[ni][nj] == 1:
                # Try perpendicular
                d = ("E" if dc >= 0 else "W") if abs(dr) >= abs(dc) else ("S" if dr >= 0 else "N")

            directions[f"{i},{j}"] = d

    # Deploy drone on highest-panic populated zone
    best_drone = None
    best_panic = 0.0
    if obs.available_resources.get("drones", 0) > 0:
        for i in range(n):
            for j in range(n):
                if obs.population_grid[i][j] > 0 and obs.panic_levels[i][j] > best_panic:
                    best_panic = obs.panic_levels[i][j]
                    best_drone = f"{i},{j}"

    return ADEMAction(
        zone_directions=directions,
        road_controls=road_controls,
        resource_allocations={},
        deploy_drone=best_drone if best_panic > 0.3 else None,
    )


# ── Episode Runner ─────────────────────────────────────────────────────────────

async def run_episode(env: ADEMEnv, client: OpenAI, task: str) -> None:
    """Run one full episode for a task, emitting mandatory structured logs."""
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    # Multi-turn conversation history: list of {role, content} dicts
    # Alternates: user (observation) → assistant (action JSON) → user → ...
    conversation_history: List[Dict[str, str]] = []

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get action from LLM (with multi-turn history)
            action = call_llm(client, obs, step, task, conversation_history)
            action_str = json.dumps(action.model_dump())

            # Step the environment
            error_msg: Optional[str] = None
            try:
                result = await env.step(action)
                obs = result.observation
                reward = float(result.reward)
                done = bool(result.done)
            except Exception as exc:
                error_msg = str(exc)[:80]
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            # Update multi-turn conversation history
            # Add the observation we acted on as "user"
            obs_summary = summarize_observation(obs, step, task)
            outcome_note = (
                f"\n[Step {step} outcome: evacuated_this_step={result.info.get('evacuated_this_step', 0)}, "
                f"casualties_this_step={result.info.get('casualties_this_step', 0)}, "
                f"total_evacuated={obs.evacuated}, total_casualties={obs.casualties}]"
            )
            conversation_history.append({"role": "user", "content": obs_summary + outcome_note})
            conversation_history.append({"role": "assistant", "content": action_str})

            if done or error_msg:
                break

        # Fetch final graded score
        try:
            score_data = await env.score()
            score = float(score_data.get("score", 0.0))
        except Exception as exc:
            print(f"[DEBUG] score fetch error: {exc}", flush=True)
            # Fallback: estimate from step rewards
            total_r = sum(rewards)
            score = min(1.0, max(0.0, 0.3 + total_r * 0.05))

        score = float(min(1.0, max(0.0, score)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error (task={task}): {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment server
    try:
        if ADEM_SERVER_URL:
            print(f"[DEBUG] Connecting to existing server: {ADEM_SERVER_URL}", flush=True)
            env = await ADEMEnv.from_server_url(ADEM_SERVER_URL)
        else:
            env = await ADEMEnv.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception as exc:
        print(f"[DEBUG] Environment startup error: {exc}", flush=True)
        return

    try:
        for task in TASKS_TO_RUN:
            await run_episode(env, client, task)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)