# adaptive_boss/client.py
"""HTTP/WebSocket client for the AdaptiveBoss OpenEnv server.

OpenEnv's canonical client base class is ``EnvClient`` (not ``HTTPEnvClient``);
``EnvClient`` already implements HTTP transport.
"""

from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ImportError:
    from core.client_types import StepResult
    from core.env_client import EnvClient

from .models import BossAction, BossObservation, BossState


class AdaptiveBossEnv(EnvClient[BossAction, BossObservation, BossState]):
    """Client for the Adaptive Boss environment.

    Example::

        client = AdaptiveBossEnv(base_url="http://localhost:8000")
        result = client.reset()
        result = client.step(BossAction(action_id=0))
        print(result.observation.boss_health, result.reward, result.done)
    """

    def _step_payload(self, action: BossAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[BossObservation]:
        return StepResult(
            observation=BossObservation(**payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> BossState:
        return BossState(**payload)
