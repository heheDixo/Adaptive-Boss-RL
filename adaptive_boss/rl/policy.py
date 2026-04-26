# adaptive_boss/rl/policy.py
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size: int = 13, n_actions: int = 3, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def actor_forward(self, state):
        return self.actor(state)

    def critic_forward(self, state):
        return self.critic(state).squeeze(-1)

    def act(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits = self.actor_forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic_forward(state)
        return action.item(), log_prob.squeeze(0), value.squeeze(0)

    def evaluate(self, states, actions):
        logits = self.actor_forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic_forward(states)
        return log_probs, entropy, values

    def action_probs(self, state):
        with torch.no_grad():
            if state.ndim == 1:
                state = state.unsqueeze(0)
            logits = self.actor_forward(state)
            return torch.softmax(logits, dim=-1).squeeze(0)
