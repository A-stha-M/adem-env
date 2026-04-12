"""Task registry split by difficulty/file for cleaner project structure."""

from .easy_building_fire import TASK as BUILDING_FIRE
from .easy_controlled_evacuation import TASK as CONTROLLED_EVACUATION
from .easy_flash_flood import TASK as FLASH_FLOOD
from .hard_hurricane_coastal import TASK as HURRICANE_COASTAL
from .hard_multi_hazard_city import TASK as MULTI_HAZARD_CITY
from .hard_panic_evacuation import TASK as PANIC_EVACUATION
from .medium_dynamic_hazard import TASK as DYNAMIC_HAZARD
from .medium_earthquake_response import TASK as EARTHQUAKE_RESPONSE
from .medium_industrial_chemical import TASK as INDUSTRIAL_CHEMICAL

TASKS = {
    "controlled_evacuation": CONTROLLED_EVACUATION,
    "flash_flood": FLASH_FLOOD,
    "building_fire": BUILDING_FIRE,
    "dynamic_hazard": DYNAMIC_HAZARD,
    "earthquake_response": EARTHQUAKE_RESPONSE,
    "industrial_chemical": INDUSTRIAL_CHEMICAL,
    "panic_evacuation": PANIC_EVACUATION,
    "hurricane_coastal": HURRICANE_COASTAL,
    "multi_hazard_city": MULTI_HAZARD_CITY,
}

__all__ = ["TASKS"]
