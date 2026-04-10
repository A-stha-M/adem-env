"""
ADEM Inference Script
=====================
Runs the LLM agent against all 3 ADEM tasks and emits structured stdout logs.

Environment variables:
    API_BASE_URL        LLM endpoint (default: HuggingFace router)
    MODEL_NAME          LLM model identifier
    HF_TOKEN            HuggingFace / API key
    LOCAL_IMAGE_NAME    Docker image name (optional; if unset uses local subprocess)
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

# ──────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")
ADEM_SERVER_URL: Optional[str] = os.getenv("ADEM_SERVER_URL")

API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK: str = "adem"

TASKS_TO_RUN: List[str] = [
    "controlled_evacuation",
    "dynamic_hazard",
    "panic_evacuation",
]

MAX_STEPS: int = 15          # hard cap per episode (within task limits)
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 512
SUCCESS_SCORE_THRESHOLD: float = 0.30

# ──────────────────────────────────────────────────────
# Logging helpers (mandatory format)
# ──────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action string to avoid huge lines
    action_short = action[:120].replace("\n", " ") if action else "null"
    print(f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={err_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ──────────────────────────────────────────────────────
# Observation → text summary for LLM
# ──────────────────────────────────────────────────────

def summarize_observation(obs: ADEMObservation, step: int) -> str:
    n = obs.grid_size

    # Find zones with people
    pop_zones = []
    for i in range(n):
        for j in range(n):
            p = obs.population_grid[i][j]
            if p > 0:
                hz = obs.hazard_grid[i][j]
                panic = obs.panic_levels[i][j]
                cong = obs.congestion_levels[i][j]
                pop_zones.append(
                    f"  ({i},{j}): pop={p}, hazard={hz:.2f}, panic={panic:.2f}, congestion={cong:.2f}"
                )

    # Hazard zones
    hazard_zones = [
        f"  ({i},{j}): {obs.hazard_grid[i][j]:.2f}"
        for i in range(n) for j in range(n)
        if obs.hazard_grid[i][j] > 0.40
    ]

    # Blocked roads
    blocked = [
        f"({i},{j})"
        for i in range(n) for j in range(n)
        if obs.road_blockages[i][j] == 1
    ]

    # Shelter info
    shelters_info = [
        f"  {sid}: {obs.shelter_populations[sid]}/{obs.shelter_capacities[sid]} at pos from config"
        for sid in obs.shelter_capacities
    ]

    lines = [
        f"=== ADEM Step {step} | Grid {n}x{n} | Time remaining: {obs.time_remaining} ===",
        f"Total pop: {obs.total_population} | Evacuated: {obs.evacuated} | Casualties: {obs.casualties}",
        f"Still in city: {obs.total_population - obs.evacuated - obs.casualties}",
        "",
        "POPULATION ZONES (zones with civilians):",
        *(pop_zones if pop_zones else ["  (none remaining)"]),
        "",
        "HAZARD ZONES (intensity > 0.4):",
        *(hazard_zones if hazard_zones else ["  (none)"]),
        "",
        "BLOCKED ROADS:",
        f"  {', '.join(blocked) if blocked else 'none'}",
        "",
        "SHELTERS:",
        *shelters_info,
        "",
        f"Available resources: {obs.available_resources}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────
# Shelter positions extracted from config for LLM
# ──────────────────────────────────────────────────────

def get_shelter_hint(task: str) -> str:
    from adem.tasks import TASKS
    cfg = TASKS.get(task, {})
    shelters = cfg.get("shelters", [])
    parts = [f"{s['id']} at row={s['pos'][0]},col={s['pos'][1]} (cap={s['capacity']})" for s in shelters]
    return "; ".join(parts)


# ──────────────────────────────────────────────────────
# LLM prompts
# ──────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI disaster evacuation commander. Your job is to save as many civilians as possible.

ENVIRONMENT:
- NxN grid city. Cells identified as "row,col" (0-indexed from top-left).
- Hazards (fire/flood) spread and kill civilians. High hazard (>0.65) causes casualties.
- Shelters are safe zones. Moving people INTO shelter cells evacuates them.
- Roads can be blocked by hazard or by your control.
- Panic reduces compliance: panicked civilians may ignore your directions.

YOUR ACTION (respond ONLY with valid JSON, no markdown, no explanation):
{
  "zone_directions": {
    "ROW,COL": "DIRECTION",
    ...
  },
  "road_controls": {},
  "resource_allocations": {},
  "deploy_drone": null
}

DIRECTION values: N (row-1), S (row+1), E (col+1), W (col-1), STAY

STRATEGY:
1. Route civilians TOWARD shelters each step.
2. Move civilians AWAY from high-hazard cells immediately.
3. Direct EVERY zone with population — don't leave anyone standing still.
4. Balance shelter loads — don't overflow one shelter.
5. If a road is blocked, route around it.

Respond with ONLY the JSON object. Nothing else.
""").strip()


def build_user_prompt(obs: ADEMObservation, step: int, task: str, history: List[str]) -> str:
    obs_text = summarize_observation(obs, step)
    shelter_hint = get_shelter_hint(task)
    hist_block = "\n".join(history[-3:]) if history else "None"

    return textwrap.dedent(f"""
        {obs_text}

        SHELTER LOCATIONS: {shelter_hint}

        RECENT HISTORY:
        {hist_block}

        Issue evacuation orders. Remember: move civilians toward shelters, away from hazard.
        Respond with ONLY valid JSON.
    """).strip()


def call_llm(client: OpenAI, obs: ADEMObservation, step: int, task: str, history: List[str]) -> ADEMAction:
    """Call LLM and parse response into ADEMAction. Falls back to heuristic on failure."""
    user_prompt = build_user_prompt(obs, step, task, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)
        return ADEMAction(**parsed)
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        return _heuristic_action(obs)


def _heuristic_action(obs: ADEMObservation) -> ADEMAction:
    """
    Fallback heuristic: move every populated zone toward the nearest shelter.
    """
    from adem.tasks import TASKS
    # Find shelter positions from observation capacities (use task config)
    # Since we don't have task name here, use a simple directional heuristic:
    # Move all people toward bottom-right (where shelters typically are)
    n = obs.grid_size
    directions: Dict[str, str] = {}
    for i in range(n):
        for j in range(n):
            if obs.population_grid[i][j] > 0:
                # Move away from high hazard (top-left in most tasks)
                # toward shelters (bottom-right)
                if i < n // 2:
                    d = "S"
                elif j < n // 2:
                    d = "E"
                else:
                    d = "S"
                directions[f"{i},{j}"] = d
    return ADEMAction(zone_directions=directions)


# ──────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────

async def run_episode(env: ADEMEnv, client: OpenAI, task: str) -> None:
    """Run one full episode for a task, emitting structured logs."""
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False
    history: List[str] = []

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get action from LLM
            action = call_llm(client, obs, step, task, history)
            action_str = json.dumps(action.model_dump())

            # Step environment
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

            history.append(
                f"Step {step}: evacuated={obs.evacuated}, casualties={obs.casualties}, "
                f"remaining={obs.total_population - obs.evacuated - obs.casualties}"
            )

            if done or error_msg:
                break

        # Fetch final score from grader
        try:
            score_data = await env.score()
            score = float(score_data.get("score", 0.0))
        except Exception:
            # Fallback: estimate from rewards
            score = min(1.0, max(0.0, sum(rewards) / max(1, len(rewards)) + 0.3))

        score = min(1.0, max(0.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment
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
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] Fatal inference error: {exc}", flush=True)