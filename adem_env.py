"""
ADEM client library — imported by inference.py.
Matches the OpenEnv client pattern from the sample inference script.

Usage:
    env = await ADEMEnv.from_docker_image(image_name)   # local Docker
    env = await ADEMEnv.from_server_url(url)             # remote / HF Space
    result = await env.reset(task="controlled_evacuation")
    result = await env.step(ADEMAction(...))
    await env.close()
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any, Dict, Optional

import httpx

from models import ADEMAction, ADEMObservation

# Re-export for inference.py convenience
__all__ = ["ADEMEnv", "ADEMAction", "ADEMObservation", "StepResult"]


class StepResult:
    """Mirrors the OpenEnv result object pattern."""
    __slots__ = ("observation", "reward", "done", "info")

    def __init__(
        self,
        observation: ADEMObservation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class ADEMEnv:
    """
    Async client for the ADEM FastAPI server.
    Supports three startup modes:
      1. from_docker_image(image_name) — pulls & runs Docker image locally
      2. from_docker_image(None)       — spawns local server subprocess
      3. from_server_url(url)          — connects to existing server (HF Space)
    """

    _DEFAULT_PORT = int(os.getenv("ADEM_PORT", "7860"))
    _HEALTH_RETRIES = 40
    _HEALTH_INTERVAL = 1.5  # seconds

    def __init__(self, base_url: str, _proc=None, _container_id: Optional[str] = None):
        self._base_url = base_url.rstrip("/")
        self._proc = _proc
        self._container_id = _container_id
        self._http = httpx.AsyncClient(timeout=60.0)

    # ──────────────────────────────────────────────
    # Factory methods
    # ──────────────────────────────────────────────

    @classmethod
    async def from_docker_image(
        cls, image_name: Optional[str], port: int = _DEFAULT_PORT
    ) -> "ADEMEnv":
        """
        Start the ADEM server:
        - If image_name is set → run Docker container.
        - If image_name is None → spawn `python server.py` subprocess.
        Then wait until /health responds and return connected client.
        """
        proc = None
        container_id = None

        if image_name:
            print(f"[DEBUG] Starting Docker container: {image_name}", flush=True)
            # Try current container port first, then legacy one for compatibility.
            last_err: Optional[Exception] = None
            for container_port in (7860, 8000):
                result = subprocess.run(
                    ["docker", "run", "-d", "-p", f"{port}:{container_port}", image_name],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    last_err = RuntimeError(f"docker run failed: {result.stderr}")
                    continue

                container_id = result.stdout.strip()
                print(
                    f"[DEBUG] Container started: {container_id[:12]} (container port {container_port})",
                    flush=True,
                )

                base_url = f"http://localhost:{port}"
                instance = cls(base_url=base_url, _proc=proc, _container_id=container_id)
                try:
                    await instance._wait_for_health()
                    return instance
                except Exception as exc:
                    last_err = exc
                    subprocess.run(
                        ["docker", "stop", container_id],
                        capture_output=True,
                        timeout=15,
                    )
                    container_id = None

            raise RuntimeError(f"Failed to start healthy Docker env: {last_err}")

        print("[DEBUG] Starting local server subprocess", flush=True)
        try:
            proc = subprocess.Popen(
                ["python", "-m", "server.app"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            # Legacy fallback for older layout.
            proc = subprocess.Popen(
                ["python", "server.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        base_url = f"http://localhost:{port}"
        instance = cls(base_url=base_url, _proc=proc, _container_id=container_id)

        # Wait for server readiness
        await instance._wait_for_health()
        return instance

    @classmethod
    async def from_server_url(cls, url: str) -> "ADEMEnv":
        """Connect to an already-running ADEM server (e.g. HF Space)."""
        instance = cls(base_url=url)
        await instance._wait_for_health()
        return instance

    # ──────────────────────────────────────────────
    # OpenEnv interface
    # ──────────────────────────────────────────────

    async def reset(self, task: str = "controlled_evacuation", seed: Optional[int] = None) -> StepResult:
        payload = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        resp = await self._http.post(f"{self._base_url}/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = ADEMObservation(**data["observation"])
        return StepResult(obs, float(data.get("reward", 0.0)), bool(data.get("done", False)), data.get("info", {}))

    async def step(self, action: ADEMAction) -> StepResult:
        resp = await self._http.post(f"{self._base_url}/step", json=action.model_dump())
        resp.raise_for_status()
        data = resp.json()
        obs = ADEMObservation(**data["observation"])
        return StepResult(obs, float(data["reward"]), bool(data["done"]), data.get("info", {}))

    async def state(self) -> Dict[str, Any]:
        resp = await self._http.get(f"{self._base_url}/state")
        resp.raise_for_status()
        return resp.json()

    async def score(self) -> Dict[str, float]:
        resp = await self._http.get(f"{self._base_url}/score")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        """Shutdown the environment and clean up resources."""
        try:
            await self._http.aclose()
        except Exception:
            pass
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
                timeout=15,
            )
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    async def _wait_for_health(self):
        url = f"{self._base_url}/health"
        for attempt in range(self._HEALTH_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        print(f"[DEBUG] Server ready at {self._base_url}", flush=True)
                        return
            except Exception:
                pass
            await asyncio.sleep(self._HEALTH_INTERVAL)
        raise TimeoutError(f"ADEM server did not become healthy at {url} after {self._HEALTH_RETRIES} attempts")