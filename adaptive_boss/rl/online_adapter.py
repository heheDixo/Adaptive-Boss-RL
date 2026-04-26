# adaptive_boss/rl/online_adapter.py
"""Online adaptation — boss fine-tunes its policy mid-fight."""
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np


class OnlineAdapter:
    """
    Performs lightweight PPO-style gradient updates during gameplay.
    Uses a small replay buffer of recent (state, action, reward) tuples.
    Updates every N steps with a conservative learning rate.
    """

    def __init__(
        self,
        policy,
        lr: float = 1e-4,        # very small — stability critical
        update_every: int = 10,   # update every 10 steps
        n_steps: int = 3,         # gradient steps per update
        buffer_size: int = 20,    # how many recent transitions to use
        gamma: float = 0.99,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.update_every = update_every
        self.n_steps = n_steps
        self.gamma = gamma

        self.buffer = deque(maxlen=buffer_size)
        self.step_count = 0
        self.update_count = 0
        self.last_loss = 0.0

    def record(self, state: np.ndarray, action: int, reward: float):
        """Call after every env.step() to record transition."""
        self.buffer.append((state.copy(), action, reward))
        self.step_count += 1

    def maybe_update(self) -> bool:
        """
        If enough steps have passed and buffer has data, run gradient update.
        Returns True if an update was performed.
        """
        if self.step_count % self.update_every != 0:
            return False
        if len(self.buffer) < self.update_every:
            return False

        self._update()
        self.update_count += 1
        return True

    def _update(self):
        """Lightweight policy gradient update using recent buffer."""
        states, actions, rewards = zip(*list(self.buffer))

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for stability
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.n_steps):
            log_probs, entropy, values = self.policy.evaluate(states_t, actions_t)

            actor_loss = -(log_probs * returns_t).mean()
            critic_loss = nn.functional.mse_loss(values, returns_t)
            entropy_bonus = entropy.mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients hard — stability critical
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.3)
            self.optimizer.step()
            total_loss += loss.item()

        self.last_loss = total_loss / self.n_steps

    def reset_episode(self):
        """Call at the start of each new episode to reset buffer."""
        self.buffer.clear()
        self.step_count = 0
