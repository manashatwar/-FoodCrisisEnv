from __future__ import annotations

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:  # pragma: no cover
    from openenv_core.env_server import create_fastapi_app

try:
    from irce.environment import FoodCrisisEnv
    from irce.models import FoodCrisisAction, FoodCrisisObservation
except ImportError:  # pragma: no cover
    from environment import FoodCrisisEnv
    from models import FoodCrisisAction, FoodCrisisObservation


def create_environment() -> FoodCrisisEnv:
    return FoodCrisisEnv()


app = create_fastapi_app(create_environment, FoodCrisisAction, FoodCrisisObservation)
