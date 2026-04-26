# adaptive_boss/train.py
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.game_logic import AdaptiveBossEnv
from rl.policy import ActorCritic
from rl.trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--model", type=str, default="models/boss_policy.pt")
    parser.add_argument("--log", type=str, default="logs/training_log.json")
    args = parser.parse_args()

    env = AdaptiveBossEnv()
    policy = ActorCritic(state_size=AdaptiveBossEnv.state_size,
                         n_actions=AdaptiveBossEnv.n_actions)
    trainer = PPOTrainer(env, policy)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    trainer.train(
        n_episodes=args.episodes,
        log_path=args.log,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
