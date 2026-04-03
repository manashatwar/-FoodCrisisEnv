from __future__ import annotations

from typing import Any

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
except ImportError:  # pragma: no cover
    from openenv_core import EnvClient
    from openenv_core.client_types import StepResult

try:
    from irce.models import IRCEAction, IRCEObservation, IRCEState
except ImportError:  # pragma: no cover
    from models import IRCEAction, IRCEObservation, IRCEState


class IRCEEnvClient(EnvClient[IRCEAction, IRCEObservation, IRCEState]):
    """Typed client for FoodCrisisEnv; package naming remains ``irce`` for compatibility."""

    def _step_payload(self, action: IRCEAction | dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(action, str):
            return IRCEAction(action_type=action).model_dump()

        if isinstance(action, dict):
            return IRCEAction.model_validate(action).model_dump()

        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[IRCEObservation]:
        observation_payload = payload.get("observation", {})
        observation = IRCEObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> IRCEState:
        return IRCEState.model_validate(payload)


class HTTPEnvClient(IRCEEnvClient):
    """Compatibility alias matching the requested client skeleton."""
