# adaptive_boss/game/game_logic.py
from collections import deque

import numpy as np

from .entities import (
    ARENA_W,
    BOSS_ACTIONS,
    MOVE_TO_INT,
    Boss,
    Player,
)

MAX_STEPS = 200


class AdaptiveBossEnv:
    """Adaptive Boss RL environment.

    State (13 dims): last 10 player moves (normalized) + boss hp + player hp + step.
    Actions: 0=attack_left, 1=attack_right, 2=reposition.
    """

    state_size = 13
    n_actions = 4  # attack_left, attack_right, reposition, defend

    def __init__(self):
        self.player = Player()
        self.boss = Boss()
        self.step_count = 0
        self.episode_count = 0
        self.episode_outcomes = deque(maxlen=200)
        self.last_player_hit = False
        self.last_boss_hit = False
        self.last_player_move = None
        self.last_boss_action = None
        self.human_mode = False

    def enable_human_mode(self):
        from game.entities import HumanPlayer
        self.player = HumanPlayer()
        self.human_mode = True

    def disable_human_mode(self):
        self.player = Player()
        self.human_mode = False

    def reset(self):
        if self.human_mode:
            from game.entities import HumanPlayer
            self.player = HumanPlayer()
        else:
            self.player = Player()
        self.boss = Boss()
        self.step_count = 0
        self.last_player_hit = False
        self.last_boss_hit = False
        self.last_player_move = None
        self.last_boss_action = None
        return self._get_state()

    def _get_state(self):
        history = self.player.move_history[-10:]
        ints = [MOVE_TO_INT[m] for m in history]
        while len(ints) < 10:
            ints.append(-1)  # -1 marks idle/padding so it cannot be confused with a real move
        # /5.0 because move IDs now range -1..4 (defend=4); padding=0.0, dodge_left=0.2,
        # dodge_right=0.4, attack=0.6, idle=0.8, defend=1.0
        norm_history = [(v + 1) / 5.0 for v in ints]
        state = np.array(
            norm_history
            + [
                self.boss.health / 100.0,
                self.player.health / 100.0,
                self.step_count / MAX_STEPS,
            ],
            dtype=np.float32,
        )
        return state

    def _win_rate(self, window=10):
        if not self.episode_outcomes:
            return 0.0
        recent = list(self.episode_outcomes)[-window:]
        return float(np.mean(recent))

    def step(self, action: int):
        boss_action = BOSS_ACTIONS[action]
        player_move = self.player.cheese_strategy()

        boss_hit = False
        player_hit = False
        prediction_correct = False
        boss_blocked = False     # boss defended a real attack
        defend_wasted = False    # boss defended but player wasn't attacking
        player_defended = (player_move == "defend")

        if player_defended:
            # Player blocks everything; no damage either way.
            pass
        elif boss_action == "defend":
            if player_move == "attack":
                boss_blocked = True
            else:
                defend_wasted = True
        elif boss_action == "attack_left":
            if player_move == "dodge_left":
                self.player.take_damage(15)
                boss_hit = True
                prediction_correct = True
            elif player_move == "attack":
                self.boss.take_damage(15)
                player_hit = True
        elif boss_action == "attack_right":
            if player_move == "dodge_right":
                self.player.take_damage(15)
                boss_hit = True
                prediction_correct = True
            elif player_move == "attack":
                self.boss.take_damage(15)
                player_hit = True
        else:  # reposition
            if self.boss.x < self.player.x:
                self.boss.x = min(self.player.x, self.boss.x + 30)
            elif self.boss.x > self.player.x:
                self.boss.x = max(self.player.x, self.boss.x - 30)

        reward = 0.0
        if boss_action == "reposition":
            reward -= 0.05  # stalling penalty — discourage reposition farming
        if boss_action == "defend":
            if boss_blocked:
                reward += 0.2  # successful block — useful but not a substitute for hitting
            else:
                reward -= 0.15  # heavy wasted-defend cost so always-defend leaks reward
        if boss_hit:
            reward += 2.0  # primary signal: damage dealt
        elif prediction_correct:
            reward += 0.5  # smaller signal for correct read without a hit (avoids double-count)
        if player_hit:
            reward -= 1.0

        # wrong_streak — boss attacked into nothing (miss OR player defend OR offside)
        if boss_action in ("attack_left", "attack_right") and not boss_hit:
            self.boss.wrong_streak += 1
            if self.boss.wrong_streak >= 3:
                reward -= 0.5
                self.boss.wrong_streak = 0
        else:
            self.boss.wrong_streak = 0

        self.player.move_history.append(player_move)
        if len(self.player.move_history) > 10:
            self.player.move_history.pop(0)

        self.step_count += 1
        self.last_player_hit = player_hit
        self.last_boss_hit = boss_hit
        self.last_player_move = player_move
        self.last_boss_action = boss_action

        done = (
            not self.player.is_alive()
            or not self.boss.is_alive()
            or self.step_count >= MAX_STEPS
        )

        if done:
            if not self.player.is_alive() and self.boss.is_alive():
                outcome = 1.0
                reward += 5.0
            elif not self.boss.is_alive() and self.player.is_alive():
                outcome = 0.0
                reward -= 5.0
            else:
                outcome = 0.5
                # Penalize timeout draws so "defend forever → draw" isn't a stable
                # equilibrium — boss must commit to actually winning the fight.
                reward -= 2.0
            outcomes = list(self.episode_outcomes)
            if len(outcomes) >= 20:
                recent10 = np.mean(outcomes[-10:])
                prior10 = np.mean(outcomes[-20:-10])
                if recent10 > prior10:
                    reward += 0.5
            self.episode_outcomes.append(outcome)
            self.episode_count += 1

        info = {
            "player_move": player_move,
            "boss_action": boss_action,
            "win_rate": self._win_rate(10),
            "boss_hit": boss_hit,
            "player_hit": player_hit,
            "prediction_correct": prediction_correct,
            "boss_blocked": boss_blocked,
            "defend_wasted": defend_wasted,
            "player_defended": player_defended,
        }

        return self._get_state(), reward, done, info
