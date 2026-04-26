"""TRL-based training pipeline for AdaptiveBossEnv.

Hackathon requirement: "A working training script using Unsloth or Hugging Face
TRL, ideally as a Colab notebook so judges can re-run it." This file fulfills
that requirement using TRL's GRPOTrainer driving a tiny GPT-2 LM as the policy.

Pipeline:
  prompt     = current 13-dim env state encoded as a short token sequence
  completion = single token in {L, R, M, D} → boss action 0..3
  reward     = AdaptiveBossEnv.step(decoded_action).reward (immediate, single-step)

The classic PyTorch PPO pipeline (train.py + rl/trainer.py) is what produced
the production v7 boss model (10 000 ep, 91% smoothed WR). This TRL pipeline
is the reproducible HF-stack alternative judges can run end-to-end in Colab.
We deliberately treat each (state, action) as a single-step bandit decision —
the env reward shaping (+2.0 hit, −1.0 hit-taken, +0.2/-0.15 defend asymmetry,
−0.05 reposition, −0.5 wrong-streak, +5/−5 terminal, −2 timeout-draw) carries
enough signal that a single-step bandit can still learn the cheese-counter map.
The longer-horizon credit assignment is what the production PPO buys you.
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game.game_logic import AdaptiveBossEnv  # noqa: E402

# ─── Vocabulary ─────────────────────────────────────────────────────────────
# Tiny domain-specific vocab. Single-character tokens keep tokenization simple
# and bound completions to one token.
ACTION_TOKENS = ["L", "R", "M", "D"]  # attack_left, attack_right, reposition, defend
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_TOKENS)}

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PROMPT_TOKENS = [
    "h", "b", "p", "s",  # delimiters: history, boss-hp, player-hp, step
    ":", ";", ",",
    "0", "1", "2", "3", "4", "5",  # bucketed values
]
VOCAB = SPECIAL_TOKENS + PROMPT_TOKENS + ACTION_TOKENS


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(VOCAB)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    return fast


# ─── State → prompt encoding ────────────────────────────────────────────────
def state_to_prompt(state: np.ndarray) -> str:
    """13-dim state → whitespace-separated single-char tokens.

    History norm ((v+1)/5) is decoded back to integer move IDs in {0..5}
    (0 = padding, 1 = dodge_left, 2 = dodge_right, 3 = attack, 4 = idle, 5 = defend).
    Health and step are bucketed to one digit (0..4) — coarse but enough to
    discriminate "early/mid/late episode" and "winning/losing the HP race".
    """
    history = state[:10]
    bhp, php, step = state[10], state[11], state[12]
    move_ids = [int(round(float(n) * 5)) for n in history]
    move_ids = [min(max(m, 0), 5) for m in move_ids]
    bhp_b = min(int(bhp * 4), 4)
    php_b = min(int(php * 4), 4)
    step_b = min(int(step * 4), 4)
    h_str = " ".join(str(m) for m in move_ids)
    return f"<bos> h : {h_str} ; b : {bhp_b} ; p : {php_b} ; s : {step_b} ;"


def _decode_action(completion: str) -> int:
    """Pick first action token in completion; fallback to random for unparseable."""
    for ch in completion:
        if ch in ACTION_TO_ID:
            return ACTION_TO_ID[ch]
    return random.randint(0, 3)


# ─── Snapshot dataset ───────────────────────────────────────────────────────
def build_snapshot_dataset(n_states: int, seed: int = 0):
    """Collect n_states env snapshots by rolling out random episodes.

    Each row: {"prompt": str, "snapshot_idx": int}. Snapshots live in a parallel
    list so the reward function can pickle.loads() the right env per row.

    States are sampled across the full episode timeline (not just resets) so the
    LM sees diverse history fillings and HP situations.
    """
    rng = random.Random(seed)
    snapshots: List[bytes] = []
    rows = []
    env = AdaptiveBossEnv()
    state = env.reset()
    for i in range(n_states):
        snap = pickle.dumps(env)
        snapshots.append(snap)
        rows.append({"prompt": state_to_prompt(state), "snapshot_idx": i})
        action = rng.randint(0, AdaptiveBossEnv.n_actions - 1)
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()
    return Dataset.from_list(rows), snapshots


# ─── Reward function ────────────────────────────────────────────────────────
class EnvSnapshotReward:
    """Reward function: decode action token, restore env from snapshot, step, return reward.

    Stateless across calls — every reward computation restores from the original
    snapshot so num_generations completions per prompt all see the same env.
    """

    __name__ = "env_step_reward"

    def __init__(self, snapshots: List[bytes]):
        self.snapshots = snapshots
        self.calls = 0

    def __call__(self, prompts, completions, snapshot_idx=None, **kwargs):
        rewards = []
        idxs = snapshot_idx if snapshot_idx is not None else [0] * len(prompts)
        for completion, idx in zip(completions, idxs):
            text = completion if isinstance(completion, str) else str(completion)
            action_id = _decode_action(text)
            env = pickle.loads(self.snapshots[idx])
            _, reward, _, _ = env.step(action_id)
            rewards.append(float(reward))
        self.calls += len(prompts)
        return rewards


# ─── Tiny GPT-2 policy ──────────────────────────────────────────────────────
def build_model(tokenizer: PreTrainedTokenizerFast) -> GPT2LMHeadModel:
    """~200K-param GPT-2: 2 layers, 64-dim embeddings, 2 attention heads.
    Big enough to learn a 13-dim → 4-action map; small enough to train in Colab.
    """
    config = GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(config)


# ─── Eval ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_episodes(model, tokenizer, n_episodes: int = 100) -> dict:
    """Roll out n_episodes with the trained LM picking argmax actions.
    Returns win-rate, draw-rate, mean reward, action distribution."""
    model.eval()
    device = next(model.parameters()).device
    env = AdaptiveBossEnv()
    wins = draws = 0
    total_reward = 0.0
    action_counts = [0, 0, 0, 0]
    for _ in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            prompt = state_to_prompt(state)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0, -1]  # next-token logits
            action_logits = torch.tensor(
                [logits[tokenizer.convert_tokens_to_ids(t)].item() for t in ACTION_TOKENS]
            )
            action = int(action_logits.argmax().item())
            action_counts[action] += 1
            state, reward, done, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
        if env.episode_outcomes and env.episode_outcomes[-1] == 1.0:
            wins += 1
        elif env.episode_outcomes and env.episode_outcomes[-1] == 0.5:
            draws += 1
    total_actions = max(sum(action_counts), 1)
    return {
        "n_episodes": n_episodes,
        "win_rate": wins / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_reward": total_reward / n_episodes,
        "action_distribution": {
            "attack_left": action_counts[0] / total_actions,
            "attack_right": action_counts[1] / total_actions,
            "reposition": action_counts[2] / total_actions,
            "defend": action_counts[3] / total_actions,
        },
    }


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_states", type=int, default=2000,
                        help="Number of (state, env-snapshot) rows in the training dataset.")
    parser.add_argument("--epochs", type=float, default=3.0,
                        help="GRPO training epochs over the snapshot dataset.")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Completions sampled per prompt by GRPO.")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./trl_output")
    parser.add_argument("--log_path", type=str, default="logs/trl_training_log.json")
    parser.add_argument("--model_path", type=str, default="models/boss_policy_trl")
    parser.add_argument("--eval_episodes", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[trl] Building tokenizer (vocab={len(VOCAB)})")
    tokenizer = build_tokenizer()

    print(f"[trl] Building tiny GPT-2 policy")
    model = build_model(tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[trl]   parameters: {n_params:,}")

    print(f"[trl] Building snapshot dataset (n_states={args.n_states})")
    dataset, snapshots = build_snapshot_dataset(args.n_states, seed=args.seed)

    reward_fn = EnvSnapshotReward(snapshots)

    print(f"[trl] Configuring GRPOTrainer")
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=1,
        max_completion_length=2,  # one action token + EOS
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        bf16=False,
        fp16=False,
        beta=0.0,  # KL=0 → no ref-model needed (we train from scratch, no pretrained ref)
        temperature=1.2,  # sample with extra entropy so GRPO sees diverse actions per prompt
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"[trl] Training (epochs={args.epochs}, "
          f"reward calls budget≈{int(args.n_states * args.epochs * args.num_generations)})")
    trainer.train()

    # ── Persist the trained policy ────────────────────────────────────────
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    print(f"[trl] Saved trained model + tokenizer to {args.model_path}")

    # ── Save the trainer log_history → plot-compatible JSON ───────────────
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    log = trainer.state.log_history
    # GRPOTrainer logs entries with 'reward' (mean reward) and 'loss'/'kl'/'reward_std'.
    # Convert to a per-step record close to the custom trainer's JSON shape so
    # generate_plots.py can render it (episode → step, boss_win_rate omitted).
    out_log = []
    for i, entry in enumerate(log):
        if "loss" not in entry and "reward" not in entry:
            continue
        out_log.append({
            "episode": i,
            "boss_win_rate": 0.0,  # placeholder; eval block below has the real number
            "avg_reward": float(entry.get("reward", 0.0)),
            "policy_loss": float(entry.get("loss", 0.0)),
            "value_loss": 0.0,
            "entropy": float(entry.get("entropy", 0.0)),
            "reward_std": float(entry.get("reward_std", 0.0)),
        })
    with open(args.log_path, "w") as f:
        json.dump(out_log, f, indent=2)
    print(f"[trl] Wrote {len(out_log)} log entries to {args.log_path}")

    # ── Evaluate the trained LM-policy ─────────────────────────────────────
    print(f"[trl] Evaluating ({args.eval_episodes} argmax episodes)")
    eval_stats = eval_episodes(model, tokenizer, n_episodes=args.eval_episodes)
    print(f"[trl] eval results:")
    print(f"        win rate: {eval_stats['win_rate']:.1%}")
    print(f"        draw rate: {eval_stats['draw_rate']:.1%}")
    print(f"        mean reward: {eval_stats['mean_reward']:.2f}")
    print(f"        actions: {eval_stats['action_distribution']}")

    eval_path = args.log_path.replace(".json", "_eval.json")
    with open(eval_path, "w") as f:
        json.dump(eval_stats, f, indent=2)
    print(f"[trl] Wrote eval to {eval_path}")


if __name__ == "__main__":
    main()
