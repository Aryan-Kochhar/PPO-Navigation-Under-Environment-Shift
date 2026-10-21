"""2D mobile-robot navigation environment (Gymnasium API).

A disc robot with unicycle kinematics must reach a goal in a 10 m x 10 m arena
containing random static blocks and moving circular obstacles. It senses with a
noisy planar LiDAR (60 beams, 240 deg) and a goal-relative vector. No map.

One env instance = one cell of the 3x3x3 shift grid (speed, density, pattern).

Set ``diagnostics=True`` (evaluation / demo only) to also compute, every step, the
quantities the failure taxonomy needs: the collision-free action count |A_free|,
lateral clearances, and contact details. Training leaves it off for speed.
"""

import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C

OBS_DIM = C.LIDAR_BEAMS + 5   # lidar, d, sin(psi), cos(psi), v_prev, w_prev


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def point_rect_dist(px, py, rects):
    """Distance from point(s) to axis-aligned rects (0 inside). Broadcasts over rects
    on the last axis: px, py shape (...,1); rects shape (R,4) -> (..., R)."""
    dx = np.maximum(np.maximum(rects[:, 0] - px, 0.0), px - rects[:, 2])
    dy = np.maximum(np.maximum(rects[:, 1] - py, 0.0), py - rects[:, 3])
    return np.hypot(dx, dy)


class NavEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(self, cell=C.CENTRE, w_c=C.W_C_DEFAULT, adaptive_risk=False,
                 diagnostics=False):
        super().__init__()
        self.speed_name, self.density_name, self.pattern = cell
        self.cell = tuple(cell)
        self.cell_id = C.cell_id(*cell)
        self.obs_speed = C.SPEED_LEVELS[self.speed_name]
        self.n_obstacles = C.DENSITY_LEVELS[self.density_name]
        self.w_c = float(w_c)
        self.adaptive_risk = adaptive_risk
        self.diagnostics = diagnostics

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(C.OBS_STACK * OBS_DIM,), dtype=np.float32)

        half = C.LIDAR_FOV / 2
        self.beam_angles = np.linspace(-half, half, C.LIDAR_BEAMS)
        self.left_beams = (self.beam_angles >= math.radians(60)) & (self.beam_angles <= math.radians(120))
        self.right_beams = (self.beam_angles <= -math.radians(60)) & (self.beam_angles >= -math.radians(120))
        self.d_norm = math.hypot(C.ARENA_SIZE, C.ARENA_SIZE)

        # candidate action set for A_free
        vv, ww = np.meshgrid(C.AFREE_V, C.AFREE_W, indexing="ij")
        self.cand_v = vv.ravel()
        self.cand_w = ww.ravel()

        self.frames = deque(maxlen=C.OBS_STACK)
        self.diag = {}

    # ------------------------------------------------------------------ #
    # episode generation
    # ------------------------------------------------------------------ #
    def _free_point(self, margin, avoid=(), avoid_dist=0.0, tries=500):
        rng = self.np_random
        for _ in range(tries):
            p = rng.uniform(margin, C.ARENA_SIZE - margin, size=2)
            if len(self.rects) and point_rect_dist(p[0], p[1], self.rects).min() < margin:
                continue
            if any(np.hypot(*(p - q)) < avoid_dist for q in avoid):
                continue
            return p
        return None

    def _generate(self):
        rng = self.np_random
        while True:
            rects = []
            for _ in range(C.N_STATIC_BLOCKS):
                w, h = rng.uniform(*C.STATIC_BLOCK_SIZE, size=2)
                x, y = rng.uniform(1.0, C.ARENA_SIZE - 1.0 - np.array([w, h]))
                rects.append([x, y, x + w, y + h])
            self.rects = np.array(rects, dtype=np.float64)

            start = self._free_point(0.6)
            if start is None:
                continue
            goal = None
            for _ in range(200):
                g = self._free_point(0.6)
                if g is not None and C.START_GOAL_DIST[0] <= np.hypot(*(g - start)) <= C.START_GOAL_DIST[1]:
                    goal = g
                    break
            if goal is None:
                continue
            break

        self.pos = start.astype(np.float64)
        self.goal = goal.astype(np.float64)
        to_goal = math.atan2(*(self.goal - self.pos)[::-1])
        self.theta = wrap_angle(to_goal + rng.uniform(-math.pi / 3, math.pi / 3))

        n = self.n_obstacles
        ob = []
        for _ in range(n):
            p = self._free_point(C.OBSTACLE_RADIUS + 0.1, avoid=(self.pos, self.goal), avoid_dist=1.5)
            if p is None:
                p = rng.uniform(1, C.ARENA_SIZE - 1, size=2)
            ob.append(p)
        self.ob_pos = np.array(ob, dtype=np.float64).reshape(n, 2)
        self.ob_head = rng.uniform(-math.pi, math.pi, size=n)
        self.ob_spd = self.obs_speed * rng.uniform(0.8, 1.2, size=n)
        self.ob_phase = rng.integers(0, C.RANDOM_WALK_INTERVAL, size=n)

    @property
    def ob_vel(self):
        return np.stack([np.cos(self.ob_head), np.sin(self.ob_head)], 1) * self.ob_spd[:, None]

    # ------------------------------------------------------------------ #
    # obstacle motion
    # ------------------------------------------------------------------ #
    def _move_obstacles(self):
        rng = self.np_random
        n = len(self.ob_pos)
        if n == 0:
            return
        if self.pattern == "random_walk":
            resample = (self.t + self.ob_phase) % C.RANDOM_WALK_INTERVAL == 0
            self.ob_head[resample] = rng.uniform(-math.pi, math.pi, size=resample.sum())
        elif self.pattern == "adversarial":
            pred = self.pos + self.v * C.ADV_LOOKAHEAD * np.array([math.cos(self.theta), math.sin(self.theta)])
            rel = pred - self.ob_pos
            near = np.hypot(rel[:, 0], rel[:, 1]) < C.ADV_RANGE
            desired = np.arctan2(rel[:, 1], rel[:, 0])
            err = wrap_angle(desired - self.ob_head)
            max_turn = C.ADV_TURN_RATE * C.DT
            self.ob_head = np.where(near, self.ob_head + np.clip(err, -max_turn, max_turn), self.ob_head)
            # far obstacles drift like a random walk
            resample = (~near) & ((self.t + self.ob_phase) % C.RANDOM_WALK_INTERVAL == 0)
            self.ob_head[resample] = rng.uniform(-math.pi, math.pi, size=resample.sum())

        new = self.ob_pos + self.ob_vel * C.DT
        r = C.OBSTACLE_RADIUS
        # bounce off arena walls
        for ax in (0, 1):
            low, high = new[:, ax] < r, new[:, ax] > C.ARENA_SIZE - r
            hit = low | high
            if hit.any():
                if ax == 0:
                    self.ob_head[hit] = wrap_angle(math.pi - self.ob_head[hit])
                else:
                    self.ob_head[hit] = wrap_angle(-self.ob_head[hit])
                new[hit, ax] = np.clip(new[hit, ax], r, C.ARENA_SIZE - r)
        # bounce off static blocks: stay put and turn around
        inside = np.zeros(n, dtype=bool)
        if len(self.rects):
            inside = point_rect_dist(new[:, 0:1], new[:, 1:2], self.rects).min(axis=1) < r
        if inside.any():
            new[inside] = self.ob_pos[inside]
            self.ob_head[inside] = wrap_angle(self.ob_head[inside] + math.pi
                                              + rng.uniform(-0.5, 0.5, size=inside.sum()))
        self.ob_pos = new

    # ------------------------------------------------------------------ #
    # sensing
    # ------------------------------------------------------------------ #
    def _raycast(self):
        ang = self.theta + self.beam_angles
        dx, dy = np.cos(ang), np.sin(ang)
        ox, oy = self.pos
        best = np.full(C.LIDAR_BEAMS, C.LIDAR_RANGE)

        # arena walls (robot is inside the box): exit distance
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 1e-9, (C.ARENA_SIZE - ox) / dx, np.where(dx < -1e-9, -ox / dx, np.inf))
            ty = np.where(dy > 1e-9, (C.ARENA_SIZE - oy) / dy, np.where(dy < -1e-9, -oy / dy, np.inf))
        best = np.minimum(best, np.minimum(tx, ty))

        # static rects, slab method  (K, R)
        if len(self.rects):
            R = self.rects
            with np.errstate(divide="ignore", invalid="ignore"):
                idx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)[:, None]
                idy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)[:, None]
                t1, t2 = (R[:, 0] - ox) / idx, (R[:, 2] - ox) / idx
                t3, t4 = (R[:, 1] - oy) / idy, (R[:, 3] - oy) / idy
            tnear = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
            tfar = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
            hit = (tnear <= tfar) & (tfar > 0)
            t = np.where(hit, np.maximum(tnear, 0.0), np.inf)
            best = np.minimum(best, t.min(axis=1))

        # dynamic circles  (K, M)
        if len(self.ob_pos):
            cx = self.ob_pos[:, 0] - ox
            cy = self.ob_pos[:, 1] - oy
            b = dx[:, None] * cx + dy[:, None] * cy
            c = cx ** 2 + cy ** 2 - C.OBSTACLE_RADIUS ** 2
            disc = b ** 2 - c
            sq = np.sqrt(np.maximum(disc, 0.0))
            t = np.where(disc >= 0, b - sq, np.inf)
            t = np.where(t > 0, t, np.where((disc >= 0) & (c < 0), 0.0, np.inf))
            best = np.minimum(best, t.min(axis=1))
        return best

    def _clearance(self):
        """m_t: clearance to nearest obstacle surface; also which obstacle."""
        r = C.ROBOT_RADIUS
        x, y = self.pos
        walls = min(x, y, C.ARENA_SIZE - x, C.ARENA_SIZE - y) - r
        stat = (point_rect_dist(x, y, self.rects).min() - r) if len(self.rects) else np.inf
        dyn, idx = np.inf, -1
        if len(self.ob_pos):
            d = np.hypot(*(self.ob_pos - self.pos).T) - r - C.OBSTACLE_RADIUS
            idx = int(np.argmin(d))
            dyn = d[idx]
        m = min(walls, stat, dyn)
        kind = "dynamic" if m == dyn else ("static" if m == stat else "wall")
        return m, kind, idx

    def _observe(self):
        self.lidar_true = self._raycast()
        noisy = self.lidar_true + self.np_random.normal(0, C.LIDAR_NOISE_STD, C.LIDAR_BEAMS)
        lid = np.clip(noisy, 0.0, C.LIDAR_RANGE) / C.LIDAR_RANGE
        rel = self.goal - self.pos
        d = math.hypot(*rel)
        psi = wrap_angle(math.atan2(rel[1], rel[0]) - self.theta)
        frame = np.concatenate([lid, [min(d / self.d_norm, 1.0), math.sin(psi), math.cos(psi),
                                      self.v / C.V_MAX, self.w / C.W_MAX]]).astype(np.float32)
        self.frames.append(frame)
        while len(self.frames) < C.OBS_STACK:
            self.frames.append(frame)
        return np.concatenate(list(self.frames))

    # ------------------------------------------------------------------ #
    # diagnostics for the failure taxonomy
    # ------------------------------------------------------------------ #
    def count_free_actions(self):
        """|A_free(s_t)|: moving candidate actions that a constant-velocity forward
        model of the *observed* obstacles predicts are collision-free for h steps
        (and, if configured, that bring the robot closer to the goal)."""
        h = C.AFREE_HORIZON
        A = len(self.cand_v)
        # robot rollouts (A, h)
        th = self.theta + np.outer(self.cand_w, np.arange(1, h + 1)) * C.DT
        px = self.pos[0] + np.cumsum(self.cand_v[:, None] * np.cos(th) * C.DT, axis=1)
        py = self.pos[1] + np.cumsum(self.cand_v[:, None] * np.sin(th) * C.DT, axis=1)
        r = C.ROBOT_RADIUS
        ok = np.ones(A, dtype=bool)
        ok &= ((px > r) & (px < C.ARENA_SIZE - r) & (py > r) & (py < C.ARENA_SIZE - r)).all(axis=1)
        if len(self.rects):
            ok &= (point_rect_dist(px[..., None], py[..., None], self.rects) > r).all(axis=(1, 2))
        if len(self.ob_pos):
            rel = self.ob_pos - self.pos
            dist = np.hypot(rel[:, 0], rel[:, 1])
            bearing = wrap_angle(np.arctan2(rel[:, 1], rel[:, 0]) - self.theta)
            seen = (dist <= C.LIDAR_RANGE) & (np.abs(bearing) <= C.LIDAR_FOV / 2)
            if seen.any():
                op, ov = self.ob_pos[seen], self.ob_vel[seen]
                steps = np.arange(1, h + 1)[:, None, None] * C.DT
                fut = op[None] + ov[None] * steps          # (h, M, 2)
                ddx = px[:, :, None] - fut[None, :, :, 0]   # (A, h, M)
                ddy = py[:, :, None] - fut[None, :, :, 1]
                ok &= (np.hypot(ddx, ddy) > r + C.OBSTACLE_RADIUS).all(axis=(1, 2))
        if C.AFREE_REQUIRE_PROGRESS:
            d_now = np.hypot(*(self.goal - self.pos))
            d_end = np.hypot(self.goal[0] - px[:, -1], self.goal[1] - py[:, -1])
            ok &= d_end < d_now - C.FREEZE_DELTA
        return int(ok.sum())

    def _contact_info(self, kind, idx):
        info = {"kind": kind, "bearing": np.nan, "closing_along_heading": False}
        if kind == "dynamic" and idx >= 0:
            rel = self.ob_pos[idx] - self.pos
            bearing = wrap_angle(math.atan2(rel[1], rel[0]) - self.theta)
            heading = np.array([math.cos(self.theta), math.sin(self.theta)])
            info["bearing"] = float(bearing)
            info["closing_along_heading"] = bool(self.ob_vel[idx] @ heading > 0)
        return info

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._generate()
        self.t = 0
        self.v = 0.0
        self.w = 0.0
        self.d_goal = math.hypot(*(self.goal - self.pos))
        self.path_length = 0.0
        self.frames.clear()
        obs = self._observe()
        self.min_clear, _, _ = self._clearance()
        self._update_diag(progress=0.0, m=self.min_clear)
        return obs, {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.v = (a[0] + 1.0) / 2.0 * C.V_MAX
        self.w = a[1] * C.W_MAX

        # obstacles move, then robot (unicycle)
        self._move_obstacles()
        self.theta = wrap_angle(self.theta + self.w * C.DT)
        old = self.pos.copy()
        self.pos = self.pos + self.v * C.DT * np.array([math.cos(self.theta), math.sin(self.theta)])
        self.path_length += math.hypot(*(self.pos - old))
        self.t += 1

        d = math.hypot(*(self.goal - self.pos))
        progress = self.d_goal - d
        self.d_goal = d
        m, kind, idx = self._clearance()
        self.min_clear = min(self.min_clear, m)

        collided = m < 0.0
        reached = (d < C.GOAL_TOLERANCE) and not collided
        timeout = self.t >= C.T_MAX and not (collided or reached)

        rho = max(0.0, C.D_SAFE - m)
        w_c = self.w_c
        if self.adaptive_risk:
            w_c *= 1.0 + C.ADAPT_KAPPA * float(np.clip((C.ADAPT_DIST - m) / C.ADAPT_DIST, 0.0, 1.0))
        reward = (C.W_GOAL * reached + C.W_PROGRESS * progress - C.W_TIME
                  - w_c * (float(collided) + C.SAFETY_LAMBDA * rho))

        obs = self._observe()
        outcome = "success" if reached else "collision" if collided else "timeout" if timeout else None
        info = {"outcome": outcome}
        if collided:
            info["contact"] = self._contact_info(kind, idx)
        self._update_diag(progress=progress, m=m)
        return obs, float(reward), bool(reached or collided), bool(timeout), info

    def _update_diag(self, progress, m):
        if not self.diagnostics:
            return
        lat_l = self.lidar_true[self.left_beams].min() - C.ROBOT_RADIUS
        lat_r = self.lidar_true[self.right_beams].min() - C.ROBOT_RADIUS
        self.diag = {
            "progress": progress, "a_free": self.count_free_actions(),
            "omega": self.w, "v": self.v, "dist_goal": self.d_goal,
            "x": self.pos[0], "y": self.pos[1], "clear_l": lat_l, "clear_r": lat_r,
            "clearance": m,
        }

    # snapshot for rendering / planning
    def state(self):
        return dict(pos=self.pos.copy(), theta=self.theta, goal=self.goal.copy(),
                    rects=self.rects.copy(), ob_pos=self.ob_pos.copy(), ob_vel=self.ob_vel.copy(),
                    lidar=self.lidar_true.copy(), beam_angles=self.beam_angles)
