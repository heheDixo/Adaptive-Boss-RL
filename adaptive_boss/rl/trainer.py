# adaptive_boss/rl/trainer.py
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPOTrainer:
    def __init__(
        self,
        env,
        policy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 512,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,  # raised from 0.03 — 5-strategy env collapsed to ATK_L 89% under 0.03
    ):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self._state = self.env.reset()
        self._episode_reward = 0.0
        self._completed_episode_rewards = []
        self._last_player_move = "idle"
        self._on_episode_end = None  # optional callback(episode_idx)

    def collect_rollouts(self, n_steps=None):
        n_steps = n_steps or self.rollout_steps
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        for _ in range(n_steps):
            state_t = torch.from_numpy(self._state).float()
            with torch.no_grad():
                action, log_prob, value = self.policy.act(state_t)
            next_state, reward, done, info = self.env.step(action)

            states.append(self._state)
            actions.append(action)
            log_probs.append(log_prob.item())
            values.append(value.item())
            rewards.append(reward)
            dones.append(done)

            self._episode_reward += reward
            self._last_player_move = info.get("player_move", self._last_player_move)
            self._state = next_state

            if done:
                self._completed_episode_rewards.append(self._episode_reward)
                if self._on_episode_end is not None:
                    self._on_episode_end(self.env.episode_count)
                self._episode_reward = 0.0
                self._state = self.env.reset()

        with torch.no_grad():
            last_state_t = torch.from_numpy(self._state).float()
            last_value = self.policy.critic_forward(last_state_t.unsqueeze(0)).item()

        advantages = np.zeros(n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n_steps)):
            next_value = last_value if t == n_steps - 1 else values[t + 1]
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)

        return {
            "states": torch.tensor(np.array(states), dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "log_probs": torch.tensor(log_probs, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "returns": torch.tensor(returns, dtype=torch.float32),
        }

    def update_policy(self, rollouts):
        states = rollouts["states"]
        actions = rollouts["actions"]
        old_log_probs = rollouts["log_probs"]
        advantages = rollouts["advantages"]
        returns = rollouts["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]
        idx = np.arange(n)
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start : start + self.batch_size]
                b_states = states[b]
                b_actions = actions[b]
                b_old_log_probs = old_log_probs[b]
                b_advantages = advantages[b]
                b_returns = returns[b]

                new_log_probs, entropy, values = self.policy.evaluate(
                    b_states, b_actions
                )
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * b_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values, b_returns)
                entropy_bonus = entropy.mean()

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy_bonus.item())

        return {
            "policy_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def _dominant_pattern(self):
        history = self.env.player.move_history
        if not history:
            return "none"
        counts = Counter(history)
        return counts.most_common(1)[0][0]

    def train(
        self,
        n_episodes: int = 1000,
        log_path: str = "logs/training_log.json",
        model_path: str = "models/boss_policy.pt",
    ):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

        log_entries = []
        last_logged_episode = -1
        last_saved_episode = -1
        best_smoothed_reward = float("-inf")
        best_model_path = model_path.replace(".pt", "_best.pt")

        # Stash the most recent update's losses so on_episode_end can attach
        # them to the corresponding per-episode log entry. Updated after each
        # PPO update; episodes that complete during a rollout share the same
        # update's losses.
        last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        def on_episode_end(ep_idx):
            win_rate = self.env._win_rate(10)
            current_reward = (
                self._completed_episode_rewards[-1]
                if self._completed_episode_rewards
                else 0.0
            )
            dominant = self._dominant_pattern()
            log_entries.append(
                {
                    "episode": ep_idx,
                    "boss_win_rate": round(win_rate, 2),
                    "avg_reward": round(current_reward, 2),
                    "dominant_player_pattern": dominant,
                    "policy_loss": round(last_losses["policy_loss"], 4),
                    "value_loss": round(last_losses["value_loss"], 4),
                    "entropy": round(last_losses["entropy"], 4),
                }
            )

        self._on_episode_end = on_episode_end

        while self.env.episode_count < n_episodes:
            rollouts = self.collect_rollouts()
            losses = self.update_policy(rollouts)
            last_losses.update(losses)

            ep = self.env.episode_count
            win_rate = self.env._win_rate(10)
            avg_reward = (
                float(np.mean(self._completed_episode_rewards[-50:]))
                if self._completed_episode_rewards
                else 0.0
            )
            dominant = self._dominant_pattern()

            if ep // 50 > last_logged_episode // 50 or last_logged_episode == -1:
                print(
                    f"Episode {ep} | Boss Win Rate: {win_rate * 100:.1f}% | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Player Pattern Detected: {dominant} dominant"
                )
                last_logged_episode = ep

            if ep // 200 > last_saved_episode // 200 and ep > 0:
                torch.save(self.policy.state_dict(), model_path)
                last_saved_episode = ep

            # Best-so-far checkpoint: track smoothed reward over last 50 completed
            # episodes, save when it improves. Guards against late-training collapse
            # (observed in the 5-strategy run where the policy peaked at ep ~1500
            # then regressed by ep 2000).
            if len(self._completed_episode_rewards) >= 100:
                smoothed = float(np.mean(self._completed_episode_rewards[-50:]))
                if smoothed > best_smoothed_reward:
                    best_smoothed_reward = smoothed
                    torch.save(self.policy.state_dict(), best_model_path)

        torch.save(self.policy.state_dict(), model_path)
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)
        print(
            f"Training complete. Saved final model to {model_path} and log to {log_path}."
        )
        if best_smoothed_reward > float("-inf"):
            print(
                f"Best-so-far model saved to {best_model_path} "
                f"(smoothed reward {best_smoothed_reward:.2f})"
            )
