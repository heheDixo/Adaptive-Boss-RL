# adaptive_boss/models.py
"""OpenEnv-compatible Pydantic models for the Adaptive Boss environment.

OpenEnv's Action / Observation / State base classes are Pydantic
``BaseModel`` subclasses (see ``openenv.core.env_server.types``); decorating
subclasses with ``@dataclass`` would bypass Pydantic validation, so we use
plain Pydantic field declarations here while keeping the field schema asked
for in the project spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server.types import Action, Observation, State
    from pydantic import Field
except ImportError:  # in-repo fallback
    from core.env_server.types import Action, Observation, State
    from pydantic import Field


# Discrete action ids used over the wire
ATTACK_LEFT = 0
ATTACK_RIGHT = 1
REPOSITION = 2
DEFEND = 3

ACTION_ID_TO_NAME = {
    ATTACK_LEFT: "attack_left",
    ATTACK_RIGHT: "attack_right",
    REPOSITION: "reposition",
    DEFEND: "defend",
}


class BossAction(Action):
    """Boss action: 0=attack_left, 1=attack_right, 2=reposition, 3=defend."""

    action_id: int = Field(..., ge=0, le=3, description="0=attack_left, 1=attack_right, 2=reposition, 3=defend")


class BossObservation(Observation):
    """Observation surfaced to the boss policy after each step."""

    player_move_history: List[int] = Field(
        default_factory=list,
        description=(
            "Last 10 player moves encoded as ints "
            "(-1=padding, 0=dodge_left, 1=dodge_right, 2=attack, 3=idle, 4=defend); "
            "padded right with -1"
        ),
    )
    boss_health: float = Field(1.0, ge=0.0, le=1.0, description="Boss HP normalized 0-1")
    player_health: float = Field(1.0, ge=0.0, le=1.0, description="Player HP normalized 0-1")
    step: int = Field(0, ge=0, description="Step index within the current episode")
    last_player_move: str = Field("", description="Player's most recent move")
    last_boss_action: str = Field("", description="Boss's most recent action")
    prediction_correct: bool = Field(False, description="Did the boss predict the player's dodge?")


class BossState(State):
    """Episode-level state surfaced through the /state endpoint."""

    episode_count: int = Field(0, ge=0, description="Total episodes elapsed on this server")
    boss_win_rate: float = Field(0.0, ge=0.0, le=1.0, description="Win rate over last 10 episodes")
    dominant_player_pattern: str = Field("none", description="Most-frequent move in current history")
