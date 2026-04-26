# adaptive_boss/server/environment.py
"""OpenEnv server-side Environment for the Adaptive Boss task."""

from __future__ import annotations

import os
import random
import sys
import uuid
from collections import Counter
from typing import Any, Optional

# Make `adaptive_boss/` importable without relying on package context
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server import Environment
except ImportError:
    from core.env_server import Environment

from game.entities import MOVE_TO_INT, BOSS_ACTIONS
from game.game_logic import AdaptiveBossEnv as GameEnv
from models import BossAction, BossObservation, BossState


class AdaptiveBossEnvironment(Environment[BossAction, BossObservation, BossState]):
    """Wraps the underlying ``GameEnv`` with OpenEnv reset/step/state hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.game = GameEnv()
        self._episode_id: Optional[str] = str(uuid.uuid4())
        self.game.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BossObservation:
        if seed is not None:
            random.seed(seed)
        self.game.reset()
        self._episode_id = episode_id or str(uuid.uuid4())
        return self._to_observation(reward=0.0, done=False, info={})

    def step(
        self,
        action: BossAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BossObservation:
        _, reward, done, info = self.game.step(action.action_id)
        if done:
            self._episode_id = str(uuid.uuid4())
        return self._to_observation(reward=float(reward), done=bool(done), info=info)

    @property
    def state(self) -> BossState:
        from collections import Counter

        history = self.game.player.move_history
        dominant = Counter(history).most_common(1)[0][0] if history else "none"
        return BossState(
            episode_id=None,
            step_count=self.game.step_count,
            episode_count=self.game.episode_count,
            boss_win_rate=self.game._win_rate(10),
            dominant_player_pattern=dominant,
        )

    def _to_observation(self, reward: float, done: bool, info: dict) -> BossObservation:
        history = list(self.game.player.move_history[-10:])
        history_ints = [MOVE_TO_INT[m] for m in history]
        while len(history_ints) < 10:
            history_ints.append(MOVE_TO_INT["idle"])
        return BossObservation(
            player_move_history=history_ints,
            boss_health=self.game.boss.health / 100.0,
            player_health=self.game.player.health / 100.0,
            step=self.game.step_count,
            last_player_move=info.get("player_move", "") if info else "",
            last_boss_action=info.get("boss_action", "") if info else "",
            prediction_correct=bool(info.get("prediction_correct", False)) if info else False,
            reward=reward,
            done=done,
        )
