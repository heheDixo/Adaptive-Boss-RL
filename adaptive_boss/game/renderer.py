# adaptive_boss/game/renderer.py
import pygame

from .entities import BOSS_ACTIONS

WINDOW_W = 900
WINDOW_H = 540  # increase to fit adaptation panel
ARENA_W = 600
BRAIN_W = 300

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 40, 40)
DARK_RED = (90, 20, 20)
BLOOD_RED = (130, 20, 20)
BLUE = (60, 120, 220)
LIGHT_BLUE = (100, 160, 255)
DARK_BLUE = (20, 60, 130)
GREEN = (60, 200, 90)
DARK_GREY = (30, 30, 35)
GREY = (90, 90, 100)
YELLOW = (240, 210, 80)
GOLD = (160, 130, 30)

FLOOR_Y = 320


class Renderer:
    def __init__(self, fps: int = 10):
        pygame.init()
        pygame.display.set_caption("Adaptive Boss — OpenEnv Hackathon")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_med = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        self.flash_color = None
        self.flash_frames = 0
        self.dmg_floats = []  # list of {text,color,x,y,frames}
        self.switch_warning_frames = 0
        self._last_history_len = -1

        # Boss lunge state — render-only, doesn't touch env
        self._lunge_remaining = 0
        self._lunge_offset_x = 0
        self._lunge_target_x = 0
        self._last_step_drawn = -1

        self._bg = self._make_background()

    def _make_background(self):
        """2.5D arena: sky + perspective floor with vanishing-point grid."""
        surf = pygame.Surface((ARENA_W, WINDOW_H))
        horizon_y = 180
        floor_front_y = FLOOR_Y  # 320
        # Trapezoid corners for the floor
        back_left, back_right = 200, ARENA_W - 200
        front_left, front_right = -40, ARENA_W + 40

        # Sky / back wall — vertical gradient down to horizon
        for y in range(horizon_y):
            t = y / horizon_y
            c = int(8 + t * 22)
            pygame.draw.line(surf, (c, c, c + 8), (0, y), (ARENA_W, y))

        # Below the floor (under the front edge) — dark void for HUD area
        pygame.draw.rect(surf, (10, 10, 14),
                         (0, floor_front_y, ARENA_W, WINDOW_H - floor_front_y))

        # Floor: trapezoid filled with depth gradient (darker at back)
        floor_depth = floor_front_y - horizon_y
        for i in range(floor_depth):
            t = i / floor_depth  # 0 at horizon, 1 at front
            y = horizon_y + i
            lx = int(back_left + (front_left - back_left) * t)
            rx = int(back_right + (front_right - back_right) * t)
            shade = int(28 + t * 30)
            pygame.draw.line(surf, (shade, shade - 4, shade - 8), (lx, y), (rx, y))

        # Perspective grid — vertical lanes converging to a single vanishing point
        vp_x = ARENA_W // 2
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            front_x = int(front_left + frac * (front_right - front_left))
            pygame.draw.line(surf, (70, 70, 80), (vp_x, horizon_y),
                             (front_x, floor_front_y), 1)

        # Horizontal grid lines — rungs spaced with perspective falloff
        for k in range(1, 6):
            t = k / 6
            # Non-linear so back rungs cluster near horizon
            t = t ** 1.6
            y = int(horizon_y + t * floor_depth)
            lx = int(back_left + (front_left - back_left) * t)
            rx = int(back_right + (front_right - back_right) * t)
            pygame.draw.line(surf, (70, 70, 80), (lx, y), (rx, y), 1)

        # Horizon line itself — bright edge
        pygame.draw.line(surf, (120, 100, 90),
                         (back_left, horizon_y),
                         (back_right, horizon_y), 1)

        # Front edge — bright "lip"
        pygame.draw.line(surf, (180, 160, 140),
                         (front_left, floor_front_y),
                         (front_right, floor_front_y), 2)

        return surf

    def _draw_shadow(self, surface, cx, feet_y, w, h):
        """Soft elliptical floor shadow under a character."""
        shadow = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 110), (0, 0, w, h))
        surface.blit(shadow, (cx - w // 2, feet_y - h // 2))

    def show_start_screen(self) -> bool:
        """Modal start screen with a PLAY button. Returns True to start, False to quit.

        Click the button, or press Space / Enter / P to start. Esc / Q quits.
        """
        title_font = pygame.font.SysFont("monospace", 56, bold=True)
        sub_font = pygame.font.SysFont("monospace", 18)
        button_font = pygame.font.SysFont("monospace", 32, bold=True)
        small_font = pygame.font.SysFont("monospace", 13)

        button_w, button_h = 240, 64
        button_rect = pygame.Rect(
            (WINDOW_W - button_w) // 2,
            300,
            button_w,
            button_h,
        )

        clock = pygame.time.Clock()
        pulse = 0.0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return False
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_p):
                        return True
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if button_rect.collidepoint(event.pos):
                        return True

            mouse_pos = pygame.mouse.get_pos()
            hover = button_rect.collidepoint(mouse_pos)

            # Background
            self.screen.fill((10, 8, 14))
            # Subtle vignette: vertical gradient
            for y in range(0, WINDOW_H, 4):
                shade = int(8 + (y / WINDOW_H) * 14)
                pygame.draw.rect(self.screen, (shade, shade // 2, shade // 2),
                                 (0, y, WINDOW_W, 4))

            # Title
            title = title_font.render("ADAPTIVE BOSS", True, BLOOD_RED)
            title_rect = title.get_rect(center=(WINDOW_W // 2, 130))
            self.screen.blit(title, title_rect)

            # Subtitle
            sub = sub_font.render(
                "The boss reads your cheese — and punishes you for it.",
                True, (200, 180, 180),
            )
            sub_rect = sub.get_rect(center=(WINDOW_W // 2, 185))
            self.screen.blit(sub, sub_rect)

            # Hackathon tag
            tag = small_font.render(
                "Meta x Scaler OpenEnv Hackathon  |  Round 2  |  Bangalore 2026",
                True, GREY,
            )
            tag_rect = tag.get_rect(center=(WINDOW_W // 2, 215))
            self.screen.blit(tag, tag_rect)

            # PLAY button
            pulse = (pulse + 0.08) % (2 * 3.14159)
            import math
            glow = int(30 + 20 * math.sin(pulse))
            base_color = (180 + glow // 2, 30, 30) if not hover else (220, 60, 60)
            pygame.draw.rect(self.screen, base_color, button_rect, border_radius=12)
            pygame.draw.rect(self.screen, (255, 200, 200), button_rect, 3, border_radius=12)
            label = button_font.render("▶  PLAY", True, WHITE)
            label_rect = label.get_rect(center=button_rect.center)
            self.screen.blit(label, label_rect)

            # Controls hint
            hints = [
                "T  cycle modes  (trained / untrained / human)",
                "R  reset fight     O  toggle online adapter     Q  quit",
                "Human mode:  Left / Right  dodge   Space  attack   D  defend",
            ]
            for i, h in enumerate(hints):
                surf = small_font.render(h, True, (170, 170, 180))
                rect = surf.get_rect(center=(WINDOW_W // 2, 410 + i * 22))
                self.screen.blit(surf, rect)

            # Footer
            footer = small_font.render(
                "Click PLAY or press Space / Enter to start",
                True, (140, 140, 150),
            )
            footer_rect = footer.get_rect(center=(WINDOW_W // 2, WINDOW_H - 25))
            self.screen.blit(footer, footer_rect)

            pygame.display.flip()
            clock.tick(30)

    def handle_events(self):
        out = {"reset": False, "quit": False, "toggle": False,
               "online_toggle": False, "human_move": None}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                out["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    out["quit"] = True
                elif event.key == pygame.K_r:
                    out["reset"] = True
                elif event.key == pygame.K_t:
                    out["toggle"] = True
                elif event.key == pygame.K_o:
                    out["online_toggle"] = True
        return out

    def on_hit(self, kind: str, env):
        if kind == "boss_hits_player":
            self.flash_color = RED
            self.flash_frames = 3
            self.dmg_floats.append(
                {
                    "text": "-15",
                    "color": RED,
                    "x": float(env.player.x),
                    "y": float(env.player.y - 30),
                    "frames": 10,
                }
            )
        elif kind == "player_hits_boss":
            self.flash_color = YELLOW
            self.flash_frames = 3
            self.dmg_floats.append(
                {
                    "text": "-15",
                    "color": YELLOW,
                    "x": float(env.boss.x),
                    "y": float(env.boss.y - 30),
                    "frames": 10,
                }
            )

    def _health_bar(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.screen, DARK_RED, (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, int(w * max(0.0, frac)), h))
        pygame.draw.rect(self.screen, WHITE, (x, y, w, h), 1)

    def _update_lunge(self, env):
        """Per-frame lunge bookkeeping: triggers on a new env step that was an attack."""
        if env.step_count != self._last_step_drawn:
            self._last_step_drawn = env.step_count
            if env.last_boss_action in ("attack_left", "attack_right"):
                self._lunge_remaining = 2
                direction = 1 if env.player.x > env.boss.x else -1
                self._lunge_target_x = direction * 28

        if self._lunge_remaining > 0:
            self._lunge_offset_x = int(self._lunge_target_x * (self._lunge_remaining / 2))
            self._lunge_remaining -= 1
        else:
            self._lunge_offset_x = 0

    def _draw_boss(self, env):
        bx = int(env.boss.x) + self._lunge_offset_x
        by = int(env.boss.y)

        # floor shadow — slightly behind / under the body
        self._draw_shadow(self.screen, bx, by + 38, 70, 14)

        # arms
        pygame.draw.rect(self.screen, (90, 10, 10), (bx - 33, by - 20, 8, 38))
        pygame.draw.rect(self.screen, (90, 10, 10), (bx + 25, by - 20, 8, 38))
        pygame.draw.rect(self.screen, (50, 5, 5), (bx - 33, by - 20, 8, 38), 1)
        pygame.draw.rect(self.screen, (50, 5, 5), (bx + 25, by - 20, 8, 38), 1)

        # body
        body = pygame.Rect(bx - 25, by - 30, 50, 60)
        pygame.draw.rect(self.screen, BLOOD_RED, body)
        pygame.draw.rect(self.screen, (60, 0, 0), body, 2)

        # horns
        pygame.draw.polygon(
            self.screen,
            (50, 5, 5),
            [(bx - 22, by - 30), (bx - 12, by - 48), (bx - 5, by - 30)],
        )
        pygame.draw.polygon(
            self.screen,
            (50, 5, 5),
            [(bx + 5, by - 30), (bx + 12, by - 48), (bx + 22, by - 30)],
        )

        # eyes (yellow with red pupil)
        pygame.draw.circle(self.screen, YELLOW, (bx - 10, by - 12), 5)
        pygame.draw.circle(self.screen, YELLOW, (bx + 10, by - 12), 5)
        pygame.draw.circle(self.screen, (200, 0, 0), (bx - 10, by - 12), 2)
        pygame.draw.circle(self.screen, (200, 0, 0), (bx + 10, by - 12), 2)

        # snarl mouth
        pygame.draw.rect(self.screen, BLACK, (bx - 10, by + 5, 20, 6))
        pygame.draw.line(self.screen, WHITE, (bx - 7, by + 5), (bx - 5, by + 11), 1)
        pygame.draw.line(self.screen, WHITE, (bx - 2, by + 5), (bx, by + 11), 1)
        pygame.draw.line(self.screen, WHITE, (bx + 3, by + 5), (bx + 5, by + 11), 1)
        pygame.draw.line(self.screen, WHITE, (bx + 7, by + 5), (bx + 9, by + 11), 1)

        # Battle axe — appears during attack actions, swings toward target
        action = env.last_boss_action
        if action == "attack_right":
            handle_start = (bx + 30, by - 2)
            handle_end = (bx + 78, by + 6)
            # handle (wood)
            pygame.draw.line(self.screen, (110, 70, 30), handle_start, handle_end, 6)
            pygame.draw.line(self.screen, (60, 35, 10), handle_start, handle_end, 1)
            # pommel cap
            pygame.draw.circle(self.screen, (60, 35, 10), handle_start, 4)
            # axe head: half-moon blade
            hx, hy = handle_end
            blade_pts = [
                (hx - 6, hy - 20),
                (hx + 22, hy - 10),
                (hx + 26, hy + 10),
                (hx - 6, hy + 20),
                (hx + 4, hy),
            ]
            pygame.draw.polygon(self.screen, (210, 215, 225), blade_pts)
            pygame.draw.polygon(self.screen, (60, 60, 80), blade_pts, 2)
            # cutting-edge highlight
            pygame.draw.line(self.screen, WHITE, (hx + 22, hy - 8), (hx + 26, hy + 8), 2)
            # swing streaks
            for i in range(3):
                pygame.draw.line(self.screen, (255, 230, 180),
                                 (bx + 20 - i * 4, by - 10 - i * 3),
                                 (bx + 28 - i * 4, by - 6 - i * 3), 2)
        elif action == "attack_left":
            handle_start = (bx - 30, by - 2)
            handle_end = (bx - 78, by + 6)
            pygame.draw.line(self.screen, (110, 70, 30), handle_start, handle_end, 6)
            pygame.draw.line(self.screen, (60, 35, 10), handle_start, handle_end, 1)
            pygame.draw.circle(self.screen, (60, 35, 10), handle_start, 4)
            hx, hy = handle_end
            blade_pts = [
                (hx + 6, hy - 20),
                (hx - 22, hy - 10),
                (hx - 26, hy + 10),
                (hx + 6, hy + 20),
                (hx - 4, hy),
            ]
            pygame.draw.polygon(self.screen, (210, 215, 225), blade_pts)
            pygame.draw.polygon(self.screen, (60, 60, 80), blade_pts, 2)
            pygame.draw.line(self.screen, WHITE, (hx - 22, hy - 8), (hx - 26, hy + 8), 2)
            for i in range(3):
                pygame.draw.line(self.screen, (255, 230, 180),
                                 (bx - 20 + i * 4, by - 10 - i * 3),
                                 (bx - 28 + i * 4, by - 6 - i * 3), 2)
        elif action == "defend":
            # crossed-arms iron barrier across the chest
            bar = pygame.Rect(bx - 30, by - 6, 60, 18)
            pygame.draw.rect(self.screen, (70, 70, 85), bar)
            pygame.draw.rect(self.screen, (30, 30, 45), bar, 2)
            # diagonal highlights to read as crossed arms
            pygame.draw.line(self.screen, (180, 180, 200), (bx - 26, by - 2), (bx + 26, by + 8), 3)
            pygame.draw.line(self.screen, (180, 180, 200), (bx + 26, by - 2), (bx - 26, by + 8), 3)
            # spikes / rivets
            for sx in (bx - 22, bx, bx + 22):
                pygame.draw.circle(self.screen, (200, 200, 215), (sx, by + 2), 2)

    def _draw_player(self, env):
        px = int(env.player.x)
        py = int(env.player.y)
        move = env.last_player_move

        # floor shadow under feet — slightly bigger because player is "closer"
        self._draw_shadow(self.screen, px, py + 42, 60, 12)

        # render onto a transparent surface so we can rotate
        size = 90
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2 + 5

        # legs
        pygame.draw.rect(surf, DARK_BLUE, (cx - 12, cy + 22, 8, 14))
        pygame.draw.rect(surf, DARK_BLUE, (cx + 4, cy + 22, 8, 14))

        # body
        body = pygame.Rect(cx - 15, cy - 10, 30, 32)
        pygame.draw.rect(surf, BLUE, body)
        pygame.draw.rect(surf, DARK_BLUE, body, 2)

        # head
        pygame.draw.circle(surf, LIGHT_BLUE, (cx, cy - 22), 12)
        pygame.draw.circle(surf, DARK_BLUE, (cx, cy - 22), 12, 2)

        # eyes
        pygame.draw.circle(surf, WHITE, (cx - 4, cy - 24), 2)
        pygame.draw.circle(surf, WHITE, (cx + 4, cy - 24), 2)
        pygame.draw.circle(surf, BLACK, (cx - 4, cy - 24), 1)
        pygame.draw.circle(surf, BLACK, (cx + 4, cy - 24), 1)

        # sword on attack
        if move == "attack":
            pygame.draw.rect(surf, GOLD, (cx + 14, cy - 4, 4, 10))
            pygame.draw.rect(surf, YELLOW, (cx + 18, cy - 2, 26, 5))
            pygame.draw.rect(surf, GOLD, (cx + 18, cy - 2, 26, 5), 1)

        # shield on defend — round shield raised in front
        if move == "defend":
            shield_rect = pygame.Rect(cx + 12, cy - 16, 14, 38)
            pygame.draw.ellipse(surf, (130, 140, 170), shield_rect)
            pygame.draw.ellipse(surf, (40, 50, 80), shield_rect, 2)
            # boss-side rivet/highlight
            pygame.draw.line(surf, WHITE, (cx + 19, cy - 10), (cx + 19, cy + 14), 1)
            pygame.draw.circle(surf, GOLD, (cx + 19, cy + 2), 2)

        angle = 0
        if move == "dodge_left":
            angle = 15
        elif move == "dodge_right":
            angle = -15

        if angle != 0:
            surf = pygame.transform.rotate(surf, angle)

        rect = surf.get_rect(center=(px, py))
        self.screen.blit(surf, rect)

    def _draw_flash(self):
        if self.flash_frames > 0 and self.flash_color is not None:
            overlay = pygame.Surface((ARENA_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((*self.flash_color, 90))
            self.screen.blit(overlay, (0, 0))
            self.flash_frames -= 1

    def _draw_dmg_floats(self):
        keep = []
        for d in self.dmg_floats:
            if d["frames"] > 0:
                txt = self.font_big.render(d["text"], True, d["color"])
                self.screen.blit(txt, (int(d["x"]) - 14, int(d["y"])))
                d["y"] -= 2.0
                d["frames"] -= 1
                keep.append(d)
        self.dmg_floats = keep

    def draw_arena(self, env, episode: int, mode: str):
        self.screen.blit(self._bg, (0, 0))
        # Lunge state advances once per env step
        self._update_lunge(env)

        # health bars
        self._health_bar(20, 15, 240, 14, env.boss.health / 100.0, RED)
        boss_label = self.font_small.render(
            f"BOSS  {env.boss.health:3d}/100", True, WHITE
        )
        self.screen.blit(boss_label, (270, 13))

        self._health_bar(20, 35, 240, 14, env.player.health / 100.0, BLUE)
        player_label = self.font_small.render(
            f"PLAYER {env.player.health:3d}/100", True, WHITE
        )
        self.screen.blit(player_label, (270, 33))

        self._draw_boss(env)
        self._draw_player(env)
        self._draw_dmg_floats()
        self._draw_flash()

        if len(env.player.move_history) >= 10:
            warning = self.font_med.render(
                "⚠ BOSS HAS YOUR PATTERN", True, (255, 80, 80)
            )
            self.screen.blit(warning, (ARENA_W // 2 - 160, 60))

        # Strategy switch flash: 3 frames every time switch_count increments
        cur_switches = getattr(env.player, "switch_count", 0)
        if cur_switches > self._last_history_len:
            self.switch_warning_frames = 3
        self._last_history_len = cur_switches
        if self.switch_warning_frames > 0:
            switch_warning = self.font_big.render(
                "⚠ PLAYER SWITCHED STRATEGY", True, (255, 200, 0)
            )
            self.screen.blit(
                switch_warning, (ARENA_W // 2 - 200, WINDOW_H // 2)
            )
            self.switch_warning_frames -= 1

        info_text = self.font_small.render(
            f"Episode {episode} | Step {env.step_count} | Mode: {mode.upper()}",
            True,
            WHITE,
        )
        self.screen.blit(info_text, (20, WINDOW_H - 25))

        if env.last_boss_action:
            act_text = self.font_small.render(
                f"Boss action: {env.last_boss_action}", True, WHITE
            )
            self.screen.blit(act_text, (20, WINDOW_H - 45))
        if env.last_player_move:
            mv_text = self.font_small.render(
                f"Player move: {env.last_player_move}", True, WHITE
            )
            self.screen.blit(mv_text, (260, WINDOW_H - 45))

        # Human-mode controls hint (mode is already shown in the bottom info line)
        if mode == "human":
            controls = self.font_small.render(
                "Left=Dodge L | Right=Dodge R | SPACE=Attack | D=Defend | T=switch mode",
                True, (200, 200, 200),
            )
            self.screen.blit(controls, (20, WINDOW_H - 65))

    def draw_outcome(self, outcome: str):
        overlay = pygame.Surface((ARENA_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        if outcome == "BOSS WINS":
            color = RED
        elif outcome == "PLAYER WINS":
            color = BLUE
        else:
            color = WHITE
        txt = self.font_huge.render(outcome, True, color)
        rect = txt.get_rect(center=(ARENA_W // 2, WINDOW_H // 2))
        self.screen.blit(txt, rect)

    def draw_brain_panel(self, env, predicted_action: str, win_rate_history, action_probs=None, mode: str = "trained"):
        panel_x = ARENA_W
        pygame.draw.rect(self.screen, DARK_GREY, (panel_x, 0, BRAIN_W, WINDOW_H))
        pygame.draw.line(self.screen, GREY, (panel_x, 0), (panel_x, WINDOW_H), 2)

        title = self.font_big.render("BOSS BRAIN", True, WHITE)
        self.screen.blit(title, (panel_x + 80, 10))

        history = env.player.move_history[-10:]
        total = max(1, len(history))
        counts = {"dodge_left": 0, "dodge_right": 0, "attack": 0}
        for m in history:
            if m in counts:
                counts[m] += 1

        chart_label = self.font_med.render("Player Pattern (last 10)", True, WHITE)
        self.screen.blit(chart_label, (panel_x + 15, 50))

        bar_x_base = panel_x + 25
        bar_y_base = 170
        bar_w = 60
        max_bar_h = 90
        labels = [("dodge_left", "L"), ("dodge_right", "R"), ("attack", "A")]
        colors = {"dodge_left": BLUE, "dodge_right": GREEN, "attack": RED}
        for i, (key, short) in enumerate(labels):
            frac = counts[key] / total
            h = int(max_bar_h * frac)
            x = bar_x_base + i * (bar_w + 20)
            pygame.draw.rect(self.screen, GREY, (x, bar_y_base - max_bar_h, bar_w, max_bar_h), 1)
            pygame.draw.rect(self.screen, colors[key], (x, bar_y_base - h, bar_w, h))
            pct = self.font_small.render(f"{int(frac * 100)}%", True, WHITE)
            self.screen.blit(pct, (x + 10, bar_y_base - max_bar_h - 18))
            lbl = self.font_small.render(short, True, WHITE)
            self.screen.blit(lbl, (x + bar_w // 2 - 4, bar_y_base + 4))

        # Pattern Lock meter
        lock_label = self.font_med.render("Pattern Lock", True, WHITE)
        self.screen.blit(lock_label, (panel_x + 15, 185))

        history_len = len(env.player.move_history)
        lock_pct = min(history_len / 10.0, 1.0)

        pygame.draw.rect(self.screen, GREY, (panel_x + 15, 205, 270, 16))

        fill_color = (
            int(100 + 155 * lock_pct),
            int(200 - 180 * lock_pct),
            50,
        )
        pygame.draw.rect(
            self.screen,
            fill_color,
            (panel_x + 15, 205, int(270 * lock_pct), 16),
        )
        pygame.draw.rect(self.screen, WHITE, (panel_x + 15, 205, 270, 16), 1)

        if lock_pct >= 1.0:
            status = self.font_small.render("PATTERN LOCKED", True, (255, 80, 80))
        else:
            status = self.font_small.render(
                f"Analyzing... {int(lock_pct * 100)}%", True, GREY
            )
        self.screen.blit(status, (panel_x + 15, 224))

        # Strategy switch indicator — switches are stochastic, so show count + rate
        switch_count = getattr(env.player, "switch_count", 0)
        switch_prob = getattr(env.player, "switch_prob", 0.0)
        if switch_count > 0:
            switch_text = self.font_small.render(
                f"Switches: {switch_count} | rate {switch_prob*100:.0f}%/step",
                True, (255, 80, 80))
        else:
            switch_text = self.font_small.render(
                f"No switch yet | rate {switch_prob*100:.0f}%/step",
                True, GREY)
        self.screen.blit(switch_text, (panel_x + 15, 245))

        pred_label = self.font_med.render("Predicts:", True, WHITE)
        self.screen.blit(pred_label, (panel_x + 15, 268))
        pred_text = self.font_med.render(predicted_action, True, YELLOW)
        self.screen.blit(pred_text, (panel_x + 100, 268))

        wr_label = self.font_med.render("Win Rate (last 20 ep)", True, WHITE)
        self.screen.blit(wr_label, (panel_x + 15, 295))

        graph_x = panel_x + 15
        graph_y = 318
        graph_w = 270
        graph_h = 60
        pygame.draw.rect(self.screen, BLACK, (graph_x, graph_y, graph_w, graph_h))
        pygame.draw.rect(self.screen, GREY, (graph_x, graph_y, graph_w, graph_h), 1)

        history_list = list(win_rate_history)
        if len(history_list) >= 2:
            n = len(history_list)
            points = []
            for i, val in enumerate(history_list):
                px = graph_x + int(i * graph_w / max(1, n - 1))
                py = graph_y + graph_h - int(val * graph_h)
                points.append((px, py))
            pygame.draw.lines(self.screen, GREEN, False, points, 2)
        elif len(history_list) == 1:
            val = history_list[0]
            py = graph_y + graph_h - int(val * graph_h)
            pygame.draw.circle(self.screen, GREEN, (graph_x + graph_w // 2, py), 3)

        cur = env._win_rate(10)
        cur_label = self.font_small.render(
            f"current: {int(cur * 100)}%", True, WHITE
        )
        self.screen.blit(cur_label, (graph_x, graph_y + graph_h + 4))

        if mode == "human":
            hint = self.font_small.render(
                "Boss is learning YOUR moves", True, (255, 200, 80)
            )
            self.screen.blit(hint, (panel_x + 15, 380))

        # Policy confidence bars (live softmax over the 3 actions)
        if action_probs is not None:
            conf_label = self.font_med.render("Policy Confidence", True, WHITE)
            self.screen.blit(conf_label, (panel_x + 15, 400))

            actions = ["ATK_L", "ATK_R", "REPOS", "DEF"]
            colors_conf = [(255, 80, 80), (80, 150, 255), (200, 200, 80), (140, 200, 220)]
            bar_y = 423
            bar_h = 11
            bar_max_w = 180

            for i, (lbl_txt, prob, color) in enumerate(zip(actions, action_probs, colors_conf)):
                y = bar_y + i * 14
                pygame.draw.rect(self.screen, GREY, (panel_x + 15, y, bar_max_w, bar_h))
                fill_w = int(bar_max_w * float(prob))
                pygame.draw.rect(self.screen, color, (panel_x + 15, y, fill_w, bar_h))
                pygame.draw.rect(self.screen, WHITE, (panel_x + 15, y, bar_max_w, bar_h), 1)
                lbl = self.font_small.render(f"{lbl_txt} {float(prob)*100:.0f}%", True, WHITE)
                self.screen.blit(lbl, (panel_x + 200, y - 1))

        # Online adaptation status
        if mode in ("trained", "human"):
            adapt_label = self.font_med.render("Online Adaptation", True, WHITE)
            self.screen.blit(adapt_label, (panel_x + 15, 485))

            update_count = getattr(self, '_adapter_updates', 0)
            loss = getattr(self, '_adapter_loss', 0.0)
            online_on = getattr(self, '_adapter_on', True)

            if not online_on:
                status = self.font_small.render(
                    "OFF (press O to enable)", True, GREY)
            elif update_count > 0:
                status = self.font_small.render(
                    f"Updates: {update_count} | Loss: {loss:.3f}",
                    True, (80, 255, 80))
            else:
                status = self.font_small.render(
                    "Waiting for data...", True, GREY)
            self.screen.blit(status, (panel_x + 15, 508))

    def flip(self):
        pygame.display.flip()
        self.clock.tick(self.fps)

    def quit(self):
        pygame.quit()
