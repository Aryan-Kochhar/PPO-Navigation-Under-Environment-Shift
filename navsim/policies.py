"""Policies with a common interface: ``reset()`` and ``act(env, obs) -> action``.

* PPOPolicy    - a trained Stable-Baselines3 PPO model (deterministic actions)
* AStarPolicy  - classical baseline: grid A* over all currently known obstacles,
                 replanned every few steps, followed with a pure-pursuit controller
* RandomPolicy - sanity check
"""

import heapq
import math

import numpy as np

from . import config as C
from .env import wrap_angle


class PPOPolicy:
    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def reset(self):
        pass

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, env, obs):
        return self.rng.uniform(-1, 1, size=2)


class AStarPolicy:
    """Geometric baseline with full knowledge of current obstacle positions.

    Occupancy grid at ``res`` metres; static blocks inflated by the robot radius,
    moving obstacles inflated by both radii plus the distance they travel in
    ``predict_s`` seconds. Replans every ``replan_every`` steps. If no path exists
    the robot stops and turns toward the goal (and may therefore freeze or time out,
    which the taxonomy then records like any other policy's failure).
    """

    def __init__(self, res=0.2, replan_every=5, margin=0.1, predict_s=0.5, lookahead=0.6):
        self.res = res
        self.replan_every = replan_every
        self.margin = margin
        self.predict_s = predict_s
        self.lookahead = lookahead
        n = int(round(C.ARENA_SIZE / res))
        self.n = n
        centres = (np.arange(n) + 0.5) * res
        self.gx, self.gy = np.meshgrid(centres, centres, indexing="ij")
        self.reset()

    def reset(self):
        self.path = None
        self.t = 0

    def _occupancy(self, env):
        r = C.ROBOT_RADIUS + self.margin
        occ = (self.gx < r) | (self.gx > C.ARENA_SIZE - r) | (self.gy < r) | (self.gy > C.ARENA_SIZE - r)
        for x0, y0, x1, y1 in env.rects:
            dx = np.maximum(np.maximum(x0 - self.gx, 0), self.gx - x1)
            dy = np.maximum(np.maximum(y0 - self.gy, 0), self.gy - y1)
            occ |= np.hypot(dx, dy) < r
        vel = env.ob_vel
        for p, v in zip(env.ob_pos, vel):
            rad = r + C.OBSTACLE_RADIUS
            # inflate along the obstacle's short-term predicted path
            for s in np.linspace(0.0, self.predict_s, 3):
                q = p + v * s
                occ |= np.hypot(self.gx - q[0], self.gy - q[1]) < rad
        return occ

    def _cell(self, p):
        return (min(self.n - 1, max(0, int(p[0] / self.res))), min(self.n - 1, max(0, int(p[1] / self.res))))

    def _plan(self, env):
        occ = self._occupancy(env)
        start, goal = self._cell(env.pos), self._cell(env.goal)
        occ[start] = False
        if occ[goal]:
            return None
        n = self.n
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        def h(c):
            dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
            return (dx + dy) + (1.414 - 2) * min(dx, dy)

        g = {start: 0.0}
        parent = {start: None}
        heap = [(h(start), 0.0, start)]
        closed = set()
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == goal:
                break
            closed.add(cur)
            for dx, dy, cost in moves:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < n and 0 <= ny < n) or occ[nx, ny]:
                    continue
                if dx and dy and (occ[cur[0] + dx, cur[1]] or occ[cur[0], cur[1] + dy]):
                    continue   # no corner cutting
                ng = gc + cost
                nb = (nx, ny)
                if ng < g.get(nb, math.inf):
                    g[nb] = ng
                    parent[nb] = cur
                    heapq.heappush(heap, (ng + h(nb), ng, nb))
        if goal not in parent:
            return None
        path, c = [], goal
        while c is not None:
            path.append(((c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res))
            c = parent[c]
        path.reverse()
        path[-1] = tuple(env.goal)
        return np.array(path)

    def act(self, env, obs):
        if self.path is None or self.t % self.replan_every == 0:
            self.path = self._plan(env)
        self.t += 1
        if self.path is None:
            # no path: stop and face the goal
            err = wrap_angle(math.atan2(*(env.goal - env.pos)[::-1]) - env.theta)
            return np.array([-1.0, np.clip(2.0 * err, -1, 1)])
        d = np.hypot(*(self.path - env.pos).T)
        i = int(np.argmin(d))
        ahead = np.nonzero(d[i:] >= self.lookahead)[0]
        target = self.path[i + ahead[0]] if len(ahead) else self.path[-1]
        err = wrap_angle(math.atan2(target[1] - env.pos[1], target[0] - env.pos[0]) - env.theta)
        w = np.clip(2.0 * err / C.W_MAX, -1.0, 1.0)
        v = C.V_MAX * max(0.0, math.cos(err)) ** 2 if abs(err) < math.pi / 2 else 0.0
        return np.array([2.0 * v / C.V_MAX - 1.0, w])


def make_policy(spec):
    """spec: ("ppo", model_path) | ("astar", None) | ("random", seed)"""
    kind, arg = spec
    if kind == "ppo":
        return PPOPolicy(arg)
    if kind == "astar":
        return AStarPolicy()
    if kind == "random":
        return RandomPolicy(arg or 0)
    raise ValueError(f"unknown policy kind {kind}")
