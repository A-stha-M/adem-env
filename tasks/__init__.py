"""Auto-discovered task registry.

Any module in this package that exports a top-level ``TASK`` dict is loaded
automatically. Task IDs are derived from filename by stripping an optional
difficulty prefix (``easy_``, ``medium_``, ``hard_``), unless a module defines
``TASK_ID`` explicitly.
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import Any, Dict


_PREFIXES = ("easy_", "medium_", "hard_")


def _task_id_from_module_name(module_name: str) -> str:
    for prefix in _PREFIXES:
        if module_name.startswith(prefix):
            return module_name[len(prefix):]
    return module_name


def _module_sort_key(module_name: str) -> tuple[int, str]:
    if module_name.startswith("easy_"):
        return (0, module_name)
    if module_name.startswith("medium_"):
        return (1, module_name)
    if module_name.startswith("hard_"):
        return (2, module_name)
    return (3, module_name)


def _discover_tasks() -> Dict[str, Dict[str, Any]]:
    task_registry: Dict[str, Dict[str, Any]] = {}

    module_names = sorted(
        (m.name for m in iter_modules(__path__) if m.name != "__init__"),
        key=_module_sort_key,
    )

    for module_name in module_names:
        module = import_module(f"{__name__}.{module_name}")
        task_obj = getattr(module, "TASK", None)
        if task_obj is None:
            continue
        if not isinstance(task_obj, dict):
            raise TypeError(f"{module_name}.TASK must be a dict")

        task_id = str(getattr(module, "TASK_ID", _task_id_from_module_name(module_name)))
        if task_id in task_registry:
            raise ValueError(f"Duplicate task id discovered: {task_id}")

        task_registry[task_id] = task_obj

    return task_registry


TASKS: Dict[str, Dict[str, Any]] = _discover_tasks()

__all__ = ["TASKS"]
