# adaptive_boss/game/entities.py
import random

ARENA_W = 600
ARENA_H = 400

MOVE_TO_INT = {"dodge_left": 0, "dodge_right": 1, "attack": 2, "idle": 3, "defend": 4}
INT_TO_MOVE = {v: k for k, v in MOVE_TO_INT.items()}

BOSS_ACTIONS = ["attack_left", "attack_right", "reposition", "defend"]

# Player cheese strategies. Each maps to a deterministic move cycle the player
# repeats until a stochastic switch picks a different one. The boss has to read
# the move history and figure out which of these distributions is currently
# running. Adding strategies here is the main lever for env innovation — bigger
# pattern space = harder pattern detection.
CHEESE_STRATEGIES = {
    "left_cheese":   ["dodge_left", "dodge_left", "attack"],
    "right_cheese":  ["dodge_right", "dodge_right", "attack"],
    "alternating":   ["dodge_left", "attack", "dodge_right", "attack"],
    "double_dodge":  ["dodge_left", "dodge_left", "dodge_right", "dodge_right", "attack"],
    "feint":         ["attack", "dodge_left", "dodge_left", "attack", "dodge_right"],
}
STRATEGY_POOL = list(CHEESE_STRATEGIES.keys())


class Player:
    def __init__(self):
        # Delegate to reset() so all attributes (including strategy/switch state)
        # are populated; otherwise game_logic.reset() — which constructs a fresh
        # Player() but never calls .reset() — leaves attributes undefined.
        self.reset()

    def reset(self):
        self.health = 100
        self.x = 300
        self.y = 200
        self.move_history = []
        self.cheese_index = 0
        self.total_moves = 0
        self.side = "center"
        self.current_strategy = random.choice(STRATEGY_POOL)
        # Per-step probability of switching strategy. Sampled once per episode
        # from a wide range so the boss can't memorize "switches happen at step N"
        # — it has to detect distribution shift from the move history itself.
        self.switch_prob = random.uniform(0.05, 0.20)
        # Per-step probability of overriding the planned cheese move with `defend`.
        # Sampled per episode in [0.10, 0.25] so the boss has to learn to read defends
        # at varying frequencies — can't memorize one rate.
        self.defend_prob = random.uniform(0.10, 0.25)
        self.has_switched = False
        self.switch_count = 0

    def cheese_strategy(self):
        # Stochastic switching: each step has switch_prob chance of swapping
        # to a different strategy. Forces the boss to learn "did the pattern
        # just change?" rather than counting steps.
        if self.total_moves > 0 and random.random() < self.switch_prob:
            others = [s for s in STRATEGY_POOL if s != self.current_strategy]
            self.current_strategy = random.choice(others)
            self.cheese_index = 0
            self.has_switched = True
            self.switch_count += 1

        sequence = CHEESE_STRATEGIES[self.current_strategy]

        move = sequence[self.cheese_index % len(sequence)]
        # Stochastic defend overrides the planned cheese move
        if random.random() < self.defend_prob:
            move = "defend"
        else:
            self.cheese_index += 1  # only advance the cheese pointer on a real cheese move
        self.total_moves += 1

        if move == "dodge_left":
            self.x = max(40, self.x - 40)
            self.side = "left"
        elif move == "dodge_right":
            self.x = min(ARENA_W - 40, self.x + 40)
            self.side = "right"

        return move

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)

    def is_alive(self):
        return self.health > 0


class HumanPlayer:
    """Human-controlled player using keyboard input."""

    def __init__(self):
        self.health = 100
        self.x = 300
        self.y = 200
        self.move_history = []
        self.side = "center"
        self.current_strategy = "human"
        self.has_switched = False
        self.switch_prob = 0.0  # never auto-switches
        self.defend_prob = 0.0  # human chooses defends explicitly
        self.switch_count = 0
        self.pending_move = "idle"
        # exposed so renderer logic that reads total_moves doesn't crash
        self.total_moves = 0
        self.cheese_index = 0

    def reset(self):
        self.health = 100
        self.x = 300
        self.y = 200
        self.move_history = []
        self.side = "center"
        self.pending_move = "idle"
        self.total_moves = 0
        self.cheese_index = 0
        self.has_switched = False
        self.switch_prob = 0.0
        self.defend_prob = 0.0
        self.switch_count = 0

    def set_move(self, move: str):
        """Called from play.py when key is pressed."""
        self.pending_move = move

    def cheese_strategy(self):
        """Returns the pending human move then resets to idle."""
        move = self.pending_move
        self.pending_move = "idle"
        if move == "dodge_left":
            self.x = max(40, self.x - 40)
            self.side = "left"
        elif move == "dodge_right":
            self.x = min(ARENA_W - 40, self.x + 40)
            self.side = "right"
        self.move_history.append(move)
        if len(self.move_history) > 10:
            self.move_history.pop(0)
        self.total_moves += 1
        self.cheese_index += 1
        return move

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)

    def is_alive(self):
        return self.health > 0


class Boss:
    def __init__(self):
        self.health = 100
        self.x = 300
        self.y = 100
        self.attack_direction = "left"
        self.wrong_streak = 0

    def reset(self):
        self.health = 100
        self.x = 300
        self.y = 100
        self.attack_direction = "left"
        self.wrong_streak = 0

    def decide_action(self, player_move_history):
        # Untrained boss: flail randomly across all 3 actions.
        # Trained boss uses the PPO policy directly (this method is unused there).
        return random.choice(BOSS_ACTIONS)

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)

    def is_alive(self):
        return self.health > 0
