"""Live demo: watch a policy drive, with the failure taxonomy computed live.

    python demo.py                                   # trained PPO at the training cell
    python demo.py --cell medium dense random_walk   # density shift (watch for freezing)
    python demo.py --cell fast medium adversarial    # speed + adversarial (collisions)
    python demo.py --policy astar                    # classical A* baseline
    python demo.py --wc 128                          # a very cautious policy from the sweep
    python demo.py --save-gif demo.gif --episodes 3  # record a GIF for the slides

Keys:  SPACE pause | N next episode | R replay episode | 1/2/3 cycle speed/density/pattern
       P cycle policy (ppo / astar / random) | L toggle lidar | +/- speed | ESC quit
"""

import argparse
import math
import os
import sys

import numpy as np

from navsim import config as C
from navsim.env import NavEnv
from navsim.evaluate import episode_seed
from navsim.metrics import EpisodeRecorder, episode_metrics
from navsim.paths import model_path
from navsim.policies import AStarPolicy, PPOPolicy, RandomPolicy

SCALE = 70                      # pixels per metre
ARENA_PX = int(C.ARENA_SIZE * SCALE)
PANEL = 340
BG, INK, INK2, MUTED = (252, 252, 251), (11, 11, 11), (82, 81, 78), (137, 135, 129)
BLOCK, GRIDC = (195, 194, 183), (225, 224, 217)
ROBOT, OBST, GOAL = (42, 120, 214), (235, 104, 52), (0, 131, 0)
RAY, WARN, BAD = (158, 197, 244), (237, 161, 0), (208, 59, 59)
# taxonomy colours match the figures; success uses navy so it never shares green with frontal/static
LABEL_COL = {"success": (22, 65, 110), "collision_from_behind": (42, 120, 214), "freezing": (235, 104, 52),
             "oscillation_near_goal": (27, 175, 122), "corridor_deadlock": (237, 161, 0),
             "timeout": (232, 123, 164), "frontal_static_collision": (0, 131, 0), "residual": MUTED}


def find_model(mode, wc, seed):
    modes = [mode] if mode else ["full", "quick", "smoke"]
    for md in modes:
        p = model_path(md, C.run_name(wc, seed))
        if os.path.exists(p):
            return p
    return None


def to_px(p):
    return int(p[0] * SCALE), int(ARENA_PX - p[1] * SCALE)


class Demo:
    def __init__(self, args):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("PPO navigation - failure-mode demo")
        self.screen = pygame.display.set_mode((ARENA_PX + PANEL, ARENA_PX))
        self.font = pygame.font.SysFont("segoeui,arial", 16)
        self.small = pygame.font.SysFont("segoeui,arial", 13)
        self.big = pygame.font.SysFont("segoeui,arial", 30, bold=True)
        self.clock = pygame.time.Clock()
        self.args = args
        self.fps = args.fps
        self.cell = list(args.cell)
        self.show_lidar = True
        self.ep = args.start_episode
        self.tally = {}
        self.frames = []

        self.policies = {"astar": AStarPolicy(), "random": RandomPolicy(0)}
        self.model_file = args.model or find_model(args.mode, args.wc, args.seed)
        if self.model_file:
            self.policies["ppo"] = PPOPolicy(self.model_file)
        self.policy_name = args.policy
        if self.policy_name == "ppo" and "ppo" not in self.policies:
            print("No trained PPO model found (run `python train.py --mode quick` first). Falling back to A*.")
            self.policy_name = "astar"

    # ------------------------------------------------------------------ #
    def new_episode(self, replay=False):
        if not replay:
            self.ep += 1
        self.env = NavEnv(cell=tuple(self.cell), diagnostics=True)
        self.obs, _ = self.env.reset(seed=episode_seed(tuple(self.cell), self.ep))
        self.policy = self.policies[self.policy_name]
        self.policy.reset()
        self.rec = EpisodeRecorder()
        self.trail = [self.env.pos.copy()]
        self.freeze_run = 0
        self.froze = False
        self.result = None
        self.result_timer = 0
        self.steps = 0

    def step(self):
        action = self.policy.act(self.env, self.obs)
        self.obs, _, term, trunc, info = self.env.step(action)
        self.rec.record(self.env, info)
        self.steps += 1
        self.trail.append(self.env.pos.copy())
        d = self.env.diag
        cond = d["progress"] < C.FREEZE_DELTA and d["a_free"] >= 1
        self.freeze_run = self.freeze_run + 1 if cond else 0
        if self.freeze_run >= C.FREEZE_N:
            self.froze = True
        if term or trunc:
            self.result = episode_metrics(self.rec, self.steps, self.env.path_length)
            lab = self.result["label"]
            self.tally[lab] = self.tally.get(lab, 0) + 1
            self.result_timer = int(1.6 * self.fps)

    # ------------------------------------------------------------------ #
    def draw(self):
        pg, s, env = self.pg, self.screen, self.env
        s.fill(BG)
        for i in range(int(C.ARENA_SIZE) + 1):
            pg.draw.line(s, GRIDC, (i * SCALE, 0), (i * SCALE, ARENA_PX), 1)
            pg.draw.line(s, GRIDC, (0, i * SCALE), (ARENA_PX, i * SCALE), 1)
        for x0, y0, x1, y1 in env.rects:
            p = to_px((x0, y1))
            pg.draw.rect(s, BLOCK, (p[0], p[1], (x1 - x0) * SCALE, (y1 - y0) * SCALE), border_radius=3)
        # goal
        g = to_px(env.goal)
        pg.draw.circle(s, GOAL, g, int(C.GOAL_TOLERANCE * SCALE), 2)
        pg.draw.circle(s, GOAL, g, 6)
        # lidar
        if self.show_lidar:
            for ang, r in zip(env.beam_angles, env.lidar_true):
                a = env.theta + ang
                end = env.pos + r * np.array([math.cos(a), math.sin(a)])
                pg.draw.line(s, RAY, to_px(env.pos), to_px(end), 1)
        # trail
        if len(self.trail) > 1:
            pg.draw.lines(s, (109, 167, 236), False, [to_px(p) for p in self.trail], 2)
        # obstacles
        for p, v in zip(env.ob_pos, env.ob_vel):
            c = to_px(p)
            pg.draw.circle(s, OBST, c, int(C.OBSTACLE_RADIUS * SCALE))
            pg.draw.line(s, INK2, c, to_px(p + v * 1.0), 2)
        # robot
        rc = to_px(env.pos)
        ring = BAD if self.froze and self.freeze_run else (WARN if self.freeze_run else ROBOT)
        pg.draw.circle(s, ROBOT, rc, int(C.ROBOT_RADIUS * SCALE))
        pg.draw.circle(s, ring, rc, int(C.ROBOT_RADIUS * SCALE) + 4, 3)
        hd = env.pos + 0.35 * np.array([math.cos(env.theta), math.sin(env.theta)])
        pg.draw.line(s, INK, rc, to_px(hd), 3)

        if self.result:
            lab = self.result["label"]
            txt = self.big.render(lab.replace("_", " ").upper(), True, LABEL_COL.get(lab, INK))
            box = txt.get_rect(center=(ARENA_PX // 2, 40))
            pg.draw.rect(s, BG, box.inflate(24, 12), border_radius=6)
            pg.draw.rect(s, LABEL_COL.get(lab, INK), box.inflate(24, 12), 2, border_radius=6)
            s.blit(txt, box)
        self.draw_panel()

    def draw_panel(self):
        pg, s, env = self.pg, self.screen, self.env
        x0 = ARENA_PX
        pg.draw.rect(s, (249, 249, 247), (x0, 0, PANEL, ARENA_PX))
        pg.draw.line(s, BLOCK, (x0, 0), (x0, ARENA_PX), 1)
        y = 14

        def line(text, font=None, color=INK, dy=22):
            nonlocal y
            s.blit((font or self.font).render(text, True, color), (x0 + 16, y))
            y += dy

        line("Failure-mode demo", self.big, dy=40)
        pol = self.policy_name.upper()
        if self.policy_name == "ppo" and self.model_file:
            pol += f"  ({os.path.basename(self.model_file)[:-4]})"
        line(f"Policy: {pol}", self.small, INK2, 18)
        train = " (training cell)" if tuple(self.cell) == C.CENTRE else "  (shifted)"
        line(f"Cell: {'-'.join(self.cell)}", self.small, INK2, 18)
        line(f"      {train}", self.small, MUTED, 18)
        line(f"Episode {self.ep}   step {self.steps}/{C.T_MAX}", self.small, INK2, 28)

        d = env.diag
        line(f"v = {d['v']:.2f} m/s    w = {d['omega']:+.2f} rad/s")
        line(f"goal distance  {d['dist_goal']:.2f} m")
        line(f"clearance m_t  {d['clearance']:.2f} m")
        line(f"|A_free| = {d['a_free']} / {len(env.cand_v)} advancing actions", dy=30)

        line("Freezing test (Eq. 5)", self.font, INK, 22)
        frac = min(1.0, self.freeze_run / C.FREEZE_N)
        pg.draw.rect(s, GRIDC, (x0 + 16, y, PANEL - 32, 12), border_radius=4)
        pg.draw.rect(s, BAD if self.freeze_run >= C.FREEZE_N else WARN,
                     (x0 + 16, y, int((PANEL - 32) * frac), 12), border_radius=4)
        y += 18
        line(f"no goal progress, clear path: {self.freeze_run}/{C.FREEZE_N} steps", self.small, INK2, 18)
        line("FREEZING EVENT this episode" if self.froze else "no freezing event yet",
             self.small, BAD if self.froze else MUTED, 30)

        line("Tally", self.font, INK, 22)
        total = sum(self.tally.values())
        for lab in ["success"] + C.FAILURE_LABELS:
            n = self.tally.get(lab, 0)
            if n or lab == "success":
                pg.draw.rect(s, LABEL_COL[lab], (x0 + 16, y + 4, 10, 10), border_radius=2)
                s.blit(self.small.render(f"{lab.replace('_', ' ')}: {n}  ({n / max(total, 1):.0%})", True, INK2),
                       (x0 + 32, y))
                y += 18
        y = ARENA_PX - 6 * 17 - 10
        for t in ["SPACE pause   N next   R replay", "1/2/3 cycle speed/density/pattern",
                  "P policy   L lidar   +/- speed   ESC quit", f"playback {self.fps} fps",
                  "ring: blue ok / yellow no goal progress /", "      red = freezing event"]:
            line(t, self.small, MUTED, 17)

    # ------------------------------------------------------------------ #
    def run(self):
        pg = self.pg
        self.new_episode()
        paused = False
        episodes_done = 0
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    return self.finish()
                if e.type == pg.KEYDOWN:
                    k = e.unicode.lower()
                    if e.key == pg.K_SPACE:
                        paused = not paused
                    elif k == "n":
                        self.new_episode()
                    elif k == "r":
                        self.new_episode(replay=True)
                    elif k in "123":
                        i = int(k) - 1
                        levels = [C.SPEEDS, C.DENSITIES, C.PATTERNS][i]
                        self.cell[i] = levels[(levels.index(self.cell[i]) + 1) % 3]
                        self.new_episode(replay=True)
                    elif k == "p":
                        names = list(self.policies)
                        self.policy_name = names[(names.index(self.policy_name) + 1) % len(names)]
                        self.new_episode(replay=True)
                    elif k == "l":
                        self.show_lidar = not self.show_lidar
                    elif k in "+=":
                        self.fps = min(120, self.fps + 5)
                    elif k == "-":
                        self.fps = max(2, self.fps - 5)
            if not paused:
                if self.result:
                    self.result_timer -= 1
                    if self.result_timer <= 0:
                        episodes_done += 1
                        if self.args.episodes and episodes_done >= self.args.episodes:
                            return self.finish()
                        self.new_episode()
                else:
                    self.step()
            self.draw()
            pg.display.flip()
            if self.args.save_gif and not paused:
                self.frames.append(pg.surfarray.array3d(self.screen).swapaxes(0, 1).copy())
            self.clock.tick(self.fps)

    def finish(self):
        if self.args.save_gif and self.frames:
            from PIL import Image
            imgs = [Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)) for f in self.frames[::2]]
            imgs[0].save(self.args.save_gif, save_all=True, append_images=imgs[1:],
                         duration=int(2000 / self.fps), loop=0)
            print(f"saved {self.args.save_gif} ({len(imgs)} frames)")
        self.pg.quit()
        print("tally:", self.tally)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="ppo", choices=["ppo", "astar", "random"])
    ap.add_argument("--mode", default=None, choices=list(C.MODES), help="which trained models to use (default: best available)")
    ap.add_argument("--model", default=None, help="explicit path to a .zip model")
    ap.add_argument("--wc", type=float, default=C.W_C_DEFAULT, help="safety weight of the model to load")
    ap.add_argument("--seed", type=int, default=0, help="training seed of the model to load")
    ap.add_argument("--cell", nargs=3, default=list(C.CENTRE), metavar=("SPEED", "DENSITY", "PATTERN"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = forever)")
    ap.add_argument("--start-episode", type=int, default=-1, help="episode index to start from (same seeds as evaluation)")
    ap.add_argument("--save-gif", default=None, help="record frames to this GIF on exit")
    a = ap.parse_args()
    if a.cell[0] not in C.SPEEDS or a.cell[1] not in C.DENSITIES or a.cell[2] not in C.PATTERNS:
        sys.exit(f"cell must be: speed in {C.SPEEDS}, density in {C.DENSITIES}, pattern in {C.PATTERNS}")
    Demo(a).run()
