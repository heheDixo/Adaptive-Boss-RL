# adaptive_boss/generate_plots.py
import json
import matplotlib.pyplot as plt
import numpy as np

with open('logs/training_log.json') as f:
    log = json.load(f)

episodes = np.array([e['episode'] for e in log])
win_rates = np.array([e['boss_win_rate'] * 100 for e in log], dtype=float)
rewards = np.array([e['avg_reward'] for e in log], dtype=float)
# Loss fields are present from v6+ runs; older logs (v1-v5) won't have them.
has_loss = all(k in log[0] for k in ("policy_loss", "value_loss", "entropy"))
if has_loss:
    policy_losses = np.array([e['policy_loss'] for e in log], dtype=float)
    value_losses = np.array([e['value_loss'] for e in log], dtype=float)
    entropies = np.array([e['entropy'] for e in log], dtype=float)


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean; first (window-1) entries shrink the window."""
    n = len(arr)
    if n == 0:
        return arr
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


# Wider window cuts the noise inherent to PPO on this 5-strategy env (per-episode
# WR is the env's own 10-ep rolling rate, so a 200-ep smooth here is effectively
# a long-horizon trend line — exactly what learning curves should show.)
SMOOTH_WIN = 200
win_rate_smooth = rolling_mean(win_rates, SMOOTH_WIN)
reward_smooth = rolling_mean(rewards, SMOOTH_WIN)

n_panels = 3 if has_loss else 2
fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
ax1, ax2 = axes[0], axes[1]
ax3 = axes[2] if has_loss else None
fig.suptitle('Adaptive Boss — PPO Learning Evidence', fontsize=16, fontweight='bold')

# Win rate plot — raw faint, smoothed bold
ax1.plot(episodes, win_rates, color='red', alpha=0.18, linewidth=1, label='Raw win rate (10-ep window)')
ax1.plot(episodes, win_rate_smooth, color='darkred', linewidth=2.5,
         label=f'Smoothed ({SMOOTH_WIN}-ep rolling mean)')
ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1.2, label='Random baseline (50%)')
ax1.fill_between(
    episodes, 50, win_rate_smooth,
    where=win_rate_smooth > 50,
    alpha=0.15, color='red', label='Improvement over random',
)
ax1.set_ylabel('Boss Win Rate (%)', fontsize=12)
ax1.set_ylim(0, 105)
ax1.set_title('Boss Win Rate vs Training Episodes', fontsize=13)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)

# Reward plot — raw faint, smoothed bold
ax2.plot(episodes, rewards, color='blue', alpha=0.18, linewidth=1, label='Raw episode reward')
ax2.plot(episodes, reward_smooth, color='navy', linewidth=2.5,
         label=f'Smoothed ({SMOOTH_WIN}-ep rolling mean)')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, label='Break-even (0)')
ax2.fill_between(
    episodes, np.minimum(reward_smooth, 0), reward_smooth,
    where=reward_smooth > 0,
    alpha=0.15, color='blue',
)
ax2.set_ylabel('Average Reward', fontsize=12)
ax2.set_title('Average Reward vs Training Episodes', fontsize=13)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
if not has_loss:
    ax2.set_xlabel('Training Episode', fontsize=12)

# Loss panel — policy / value / entropy on twin axes
if has_loss:
    policy_smooth = rolling_mean(policy_losses, SMOOTH_WIN)
    value_smooth = rolling_mean(value_losses, SMOOTH_WIN)
    entropy_smooth = rolling_mean(entropies, SMOOTH_WIN)

    ax3.plot(episodes, policy_smooth, color='darkorange', linewidth=2.2, label='Policy loss (smoothed)')
    ax3.plot(episodes, value_smooth, color='purple', linewidth=2.2, label='Value loss (smoothed)')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1.0)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_xlabel('Training Episode', fontsize=12)
    ax3.set_title('PPO Losses & Policy Entropy', fontsize=13)
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    ax3b.plot(episodes, entropy_smooth, color='teal', linewidth=2.2,
              linestyle='--', label='Entropy (smoothed)')
    ax3b.set_ylabel('Entropy', fontsize=12, color='teal')
    ax3b.tick_params(axis='y', labelcolor='teal')

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('logs/training_curve.png', dpi=150, bbox_inches='tight')
print('Saved training_curve.png')

print(f'\n=== TRAINING SUMMARY ===')
print(f'Total episodes: {int(episodes.max())}')
print(f'Smoothed start win rate: {win_rate_smooth[min(SMOOTH_WIN, len(win_rate_smooth)-1)]:.0f}%')
print(f'Smoothed final win rate: {win_rate_smooth[-1]:.0f}%')
ep80 = next((int(e) for e, w in zip(episodes, win_rate_smooth) if w >= 80), 'N/A')
print(f'Episodes for smoothed curve to reach 80%: {ep80}')
print(f'Smoothed start reward: {reward_smooth[min(SMOOTH_WIN, len(reward_smooth)-1)]:.2f}')
print(f'Smoothed final reward: {reward_smooth[-1]:.2f}')
delta = reward_smooth[-1] - reward_smooth[min(SMOOTH_WIN, len(reward_smooth)-1)]
print(f'Smoothed reward delta: {delta:+.2f}')
