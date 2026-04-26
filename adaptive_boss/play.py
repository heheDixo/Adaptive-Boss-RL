# adaptive_boss/play.py
import copy
import os
import random
import sys
from collections import deque

import pygame
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.entities import BOSS_ACTIONS, HumanPlayer
from game.game_logic import AdaptiveBossEnv
from game.renderer import Renderer
from rl.online_adapter import OnlineAdapter
from rl.policy import ActorCritic

MODEL_PATH = "models/boss_policy.pt"
OUTCOME_PAUSE_FRAMES = 20  # 2 seconds at 10fps


def load_policy():
    policy = ActorCritic(state_size=AdaptiveBossEnv.state_size,
                         n_actions=AdaptiveBossEnv.n_actions)
    if not os.path.exists(MODEL_PATH):
        return policy, False
    try:
        policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        policy.eval()
        return policy, True
    except RuntimeError as e:
        # Shape mismatch — saved policy was trained for a different action/state size.
        # Most common cause: action space changed. Fall back to untrained.
        print(f"[load_policy] saved model incompatible ({e.__class__.__name__}): "
              "falling back to untrained boss. Retrain via `python train.py`.")
        return policy, False


def main():
    policy, has_trained = load_policy()
    mode = "trained" if has_trained else "untrained"

    # Snapshot original weights — frozen policy used for actual decisions
    original_state_dict = copy.deepcopy(policy.state_dict())

    # Display policy: a live clone the adapter mutates. Drives the BOSS BRAIN
    # confidence bars and the loss/updates readout, but does NOT pick actions.
    # Frozen `policy` keeps the boss's play strength intact.
    display_policy = ActorCritic(state_size=AdaptiveBossEnv.state_size,
                                 n_actions=AdaptiveBossEnv.n_actions)
    display_policy.load_state_dict(copy.deepcopy(policy.state_dict()))

    adapter = OnlineAdapter(
        policy=display_policy,
        lr=1e-4,
        update_every=10,
        n_steps=3,
        buffer_size=20,
    )
    online_adapt = True  # toggle with O key

    env = AdaptiveBossEnv()
    state = env.reset()
    renderer = Renderer(fps=10)

    if not renderer.show_start_screen():
        renderer.quit()
        return

    win_rate_history = deque(maxlen=20)
    pause_frames = 0
    outcome = None
    episode_idx = 1

    running = True
    while running:
        events = renderer.handle_events()
        if events["quit"]:
            running = False
            break
        if events["reset"]:
            if mode == "human":
                env.enable_human_mode()
            else:
                env.disable_human_mode()
            state = env.reset()
            pause_frames = 0
            outcome = None
        if events["toggle"]:
            # Cycle: trained → untrained → human → trained
            if mode == "trained":
                mode = "untrained"
                env.disable_human_mode()
            elif mode == "untrained":
                mode = "human"
                env.enable_human_mode()
            else:
                mode = "trained"
                env.disable_human_mode()
            state = env.reset()
            adapter.reset_episode()
            pause_frames = 0
            outcome = None
        if events.get("online_toggle"):
            online_adapt = not online_adapt
            print(f"Online adaptation: {'ON' if online_adapt else 'OFF'}")

        # Human keyboard input — read each frame in human mode
        if mode == "human" and isinstance(env.player, HumanPlayer):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                env.player.set_move("dodge_left")
            elif keys[pygame.K_RIGHT]:
                env.player.set_move("dodge_right")
            elif keys[pygame.K_SPACE]:
                env.player.set_move("attack")
            elif keys[pygame.K_d]:
                env.player.set_move("defend")
            else:
                env.player.set_move("idle")

        if pause_frames > 0:
            pause_frames -= 1
            if pause_frames == 0:
                state = env.reset()
                outcome = None
        else:
            # Boss uses trained policy in both "trained" and "human" modes —
            # but only if we actually loaded trained weights. argmax of a
            # randomly-initialized net is deterministic per state and biased
            # toward whichever output has the largest random init, so without
            # this guard the boss locks onto one action (e.g. always defends).
            if (mode == "trained" or mode == "human") and has_trained:
                state_t = torch.from_numpy(state).float()
                with torch.no_grad():
                    probs = policy.action_probs(state_t)
                action = int(torch.argmax(probs).item())
            else:
                action = random.randint(0, AdaptiveBossEnv.n_actions - 1)

            state, reward, done, info = env.step(action)

            # Record transition for online adaptation
            if online_adapt and mode in ("trained", "human"):
                adapter.record(state, action, reward)
                adapter.maybe_update()

            if info.get("boss_hit"):
                renderer.on_hit("boss_hits_player", env)
            if info.get("player_hit"):
                renderer.on_hit("player_hits_boss", env)

            if done:
                win_rate_history.append(env._win_rate(10))
                episode_idx = env.episode_count + 1
                if not env.player.is_alive() and env.boss.is_alive():
                    outcome = "BOSS WINS"
                elif not env.boss.is_alive() and env.player.is_alive():
                    outcome = "PLAYER WINS"
                else:
                    outcome = "DRAW"
                pause_frames = OUTCOME_PAUSE_FRAMES
                # Reset adapter buffer but KEEP adapted weights — boss carries
                # learning across episodes. To reset weights instead, uncomment:
                # policy.load_state_dict(copy.deepcopy(original_state_dict))
                adapter.reset_episode()

        state_t = torch.from_numpy(state).float()
        with torch.no_grad():
            # Display probs come from the live-adapting clone so the bars
            # actually shift; the boss still acts via the frozen `policy`.
            display_probs = display_policy.action_probs(state_t)
        predicted_action = BOSS_ACTIONS[int(torch.argmax(display_probs).item())]
        if mode == "untrained" or not has_trained:
            predicted_action = "random"

        renderer._adapter_updates = adapter.update_count
        renderer._adapter_loss = adapter.last_loss
        renderer._adapter_on = online_adapt

        renderer.draw_arena(env, episode_idx, mode)
        if outcome is not None and pause_frames > 0:
            renderer.draw_outcome(outcome)
        renderer.draw_brain_panel(
            env,
            predicted_action,
            win_rate_history,
            action_probs=display_probs.numpy() if mode != "untrained" else None,
            mode=mode,
        )
        renderer.flip()

    renderer.quit()


if __name__ == "__main__":
    main()
