# adaptive_boss/__init__.py
"""Adaptive Boss — OpenEnv-compatible RL environment.

Public API:
    AdaptiveBossEnv — HTTP/WS client for the running env server
    BossAction      — discrete boss action (0=attack_left, 1=attack_right, 2=reposition)
    BossObservation — typed observation returned by the env
    BossState       — episode-level state via the /state endpoint
"""

from .client import AdaptiveBossEnv
from .models import BossAction, BossObservation, BossState

__all__ = [
    "AdaptiveBossEnv",
    "BossAction",
    "BossObservation",
    "BossState",
]
