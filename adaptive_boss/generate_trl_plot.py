"""Plot the TRL training curve from logs/trl_training_log.json.

GRPOTrainer logs per-training-step (not per-episode), so the panels here are:
  1. Mean reward per training step (with band for ±1 std)
  2. Policy loss per training step
  3. Entropy per training step

The final argmax-eval stats are loaded from logs/trl_training_log_eval.json
and shown in a small annotation box.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("logs/trl_training_log.json")
EVAL_PATH = Path("logs/trl_training_log_eval.json")
OUT_PATH = Path("logs/trl_training_curve.png")


def main():
    if not LOG_PATH.exists():
        raise SystemExit(f"missing {LOG_PATH} — run train_trl.py first")
    with open(LOG_PATH) as f:
        log = json.load(f)
    if not log:
        raise SystemExit(f"empty log at {LOG_PATH}")

    steps = np.arange(len(log))
    rewards = np.array([e.get("avg_reward", 0.0) for e in log], dtype=float)
    reward_std = np.array([e.get("reward_std", 0.0) for e in log], dtype=float)
    losses = np.array([e.get("policy_loss", 0.0) for e in log], dtype=float)
    entropies = np.array([e.get("entropy", 0.0) for e in log], dtype=float)

    eval_stats = None
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            eval_stats = json.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("Adaptive Boss — TRL (GRPO) Learning Evidence", fontsize=15, fontweight="bold")
    ax1, ax2, ax3 = axes

    # ── Reward ──────────────────────────────────────────────────────────
    ax1.plot(steps, rewards, color="navy", linewidth=2, label="Mean reward per training step")
    ax1.fill_between(steps, rewards - reward_std, rewards + reward_std,
                     color="navy", alpha=0.15, label="±1σ across completions")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax1.set_ylabel("Mean step reward", fontsize=12)
    ax1.set_title("GRPO reward — single-step env return per (state, action) sample", fontsize=12)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Policy loss ─────────────────────────────────────────────────────
    ax2.plot(steps, losses, color="darkorange", linewidth=2, label="Policy loss")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("GRPO policy loss", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Entropy ─────────────────────────────────────────────────────────
    ax3.plot(steps, entropies, color="teal", linewidth=2, label="Token-distribution entropy")
    ax3.set_ylabel("Entropy (nats)", fontsize=12)
    ax3.set_xlabel("Training step", fontsize=12)
    ax3.set_title("Policy entropy", fontsize=12)
    ax3.legend(fontsize=10, loc="upper right")
    ax3.grid(True, alpha=0.3)

    if eval_stats:
        wr = eval_stats["win_rate"] * 100
        dr = eval_stats["draw_rate"] * 100
        mr = eval_stats["mean_reward"]
        n_ep = eval_stats["n_episodes"]
        text = (f"Argmax eval ({n_ep} ep)\n"
                f"  Win rate: {wr:.1f}%\n"
                f"  Draw rate: {dr:.1f}%\n"
                f"  Mean ep reward: {mr:.2f}")
        ax1.text(0.02, 0.98, text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                           edgecolor="goldenrod", alpha=0.9))

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")

    print("\n=== TRL TRAINING SUMMARY ===")
    print(f"Training steps logged: {len(log)}")
    print(f"Reward: {rewards[0]:+.3f} → {rewards[-1]:+.3f} (Δ {rewards[-1]-rewards[0]:+.3f})")
    if eval_stats:
        print(f"Argmax win rate ({eval_stats['n_episodes']} ep): {eval_stats['win_rate']:.1%}")
        print(f"Mean episode reward: {eval_stats['mean_reward']:.2f}")
        print(f"Action distribution: {eval_stats['action_distribution']}")


if __name__ == "__main__":
    main()
