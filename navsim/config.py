"""Pre-registered constants for the whole project.

Everything the paper calls "fixed before any experiment runs" lives here: the
simulator geometry, the shift grid, the reward weights, the taxonomy thresholds
and the experiment sizes for each run mode. Every script imports from this file,
so Study A, Study B and the baselines are guaranteed to use identical metric code
and thresholds (the "interface contract" in Section IV-D of the paper).

Do NOT change a threshold after looking at results. If you must change one, change
it here, write down why, and re-run everything.
"""

import math
from itertools import product

# --------------------------------------------------------------------------- #
# Simulator geometry (paper Table III)
# --------------------------------------------------------------------------- #
DT = 0.1                    # control period (s)  -> 10 Hz
V_MAX = 0.5                 # max linear speed (m/s)
W_MAX = 1.0                 # max angular speed (rad/s)
ROBOT_RADIUS = 0.2          # r_r (m)
OBSTACLE_RADIUS = 0.25      # r_o (m), dynamic obstacles
ARENA_SIZE = 10.0           # square arena side (m)
N_STATIC_BLOCKS = 4         # random axis-aligned static blocks per episode
STATIC_BLOCK_SIZE = (0.6, 1.8)
GOAL_TOLERANCE = 0.3        # goal reached when d_t < this (m)
START_GOAL_DIST = (6.0, 9.0)

LIDAR_BEAMS = 60            # K
LIDAR_FOV = math.radians(240.0)   # Phi
LIDAR_RANGE = 4.0           # max range (m)
LIDAR_NOISE_STD = 0.02      # additive Gaussian noise (m)
OBS_STACK = 4               # H
T_MAX = 500                 # episode step limit

# --------------------------------------------------------------------------- #
# Reward, Eq. (1):  r = w_g*1[goal] + w_p*progress - w_t - w_c*(1[coll] + lam*rho)
# --------------------------------------------------------------------------- #
W_GOAL = 10.0
W_PROGRESS = 5.0            # per metre of progress toward the goal
W_TIME = 0.01               # per-step time cost
SAFETY_LAMBDA = 1.0         # fixed ratio between proximity shaping and collision
D_SAFE = 0.3                # d_safe (m)

W_C_DEFAULT = 8.0           # "PPO fixed reward" baseline (task-tuned default)
W_C_SAFE = 32.0             # "PPO fixed safety reward" baseline (elevated, 4x)

# Adaptive risk weighting baseline: w_c scaled online by clearance m_t
ADAPT_KAPPA = 3.0           # w_c_eff = w_c * (1 + kappa * clip((d_adapt - m)/d_adapt, 0, 1))
ADAPT_DIST = 1.0

# --------------------------------------------------------------------------- #
# Pre-registered 3x3x3 shift grid (paper Table II). Training = centre cell.
# --------------------------------------------------------------------------- #
SPEED_LEVELS = {"slow": 0.15, "medium": 0.30, "fast": 0.50}        # m/s
DENSITY_LEVELS = {"sparse": 4, "medium": 8, "dense": 14}           # dynamic obstacles
PATTERN_LEVELS = ["linear", "random_walk", "adversarial"]
SPEEDS = list(SPEED_LEVELS)
DENSITIES = list(DENSITY_LEVELS)
PATTERNS = PATTERN_LEVELS

RANDOM_WALK_INTERVAL = 20   # steps between heading resamples (2 s)
ADV_RANGE = 4.0             # adversarial obstacles chase only within this range (m)
ADV_LOOKAHEAD = 1.0         # predict robot position this far ahead (s)
ADV_TURN_RATE = 1.0         # max heading change of an adversarial obstacle (rad/s)


def cell_id(speed, density, pattern):
    return f"{speed}-{density}-{pattern}"


CENTRE = ("medium", "medium", "random_walk")
CENTRE_ID = cell_id(*CENTRE)
ALL_CELLS = list(product(SPEEDS, DENSITIES, PATTERNS))   # 27 cells

# --------------------------------------------------------------------------- #
# Failure taxonomy thresholds (paper Section IV-A and VI). FROZEN.
# --------------------------------------------------------------------------- #
# Freezing, Eq. (5)
FREEZE_DELTA = 0.1 * V_MAX * DT     # delta = 5e-3 m per step
FREEZE_N = 20                       # window N (steps) = 2.0 s
AFREE_HORIZON = 10                  # h: forward-simulation horizon (steps) = 1.0 s
# Candidate action set A_Delta: moving actions only (v > 0). An action counts as
# "free" if the constant-velocity forward model predicts no collision over h steps
# AND it reduces goal distance by at least FREEZE_DELTA, i.e. the robot "could
# have advanced" (paper Section VI-A).
AFREE_V = [0.5 * V_MAX, 1.0 * V_MAX]
AFREE_W = [-1.0 * W_MAX, -0.5 * W_MAX, 0.0, 0.5 * W_MAX, 1.0 * W_MAX]
AFREE_REQUIRE_PROGRESS = True

# Oscillation-near-goal
OSC_RADIUS = 1.0            # R_g (m)
OSC_MIN_FLIPS = 6           # M sign changes of omega...
OSC_WINDOW = 20             # ...within N_o steps
OSC_DEADBAND = 0.05 * W_MAX # |omega| below this counts as zero (no sign)

# Corridor-deadlock
DEADLOCK_LATERAL = 0.35     # lateral clearance threshold, each side (m)
DEADLOCK_WINDOW = 20        # window (steps)
DEADLOCK_EPS = 0.05         # epsilon_d: net displacement over window (m)

# Residual class: if residual > 10% of failures in a cell, the taxonomy
# under-covers the failure space and this is reported as a negative result.
RESIDUAL_LIMIT = 0.10

# Sensitivity analysis grid (paper Section VI-B)
SENS_DELTA_FRACS = [0.05, 0.10, 0.15, 0.20]     # x V_MAX*DT
SENS_N = [10, 20, 30, 40]

# Ordered tests, first match wins. "frontal_static_collision" was added on
# 2026-10-21 after a pilot run (one 400k-step policy, 120 episodes) showed that
# head-on contacts and contacts with walls/static blocks matched none of the
# original five tests (45 of 67 failures fell into the residual class). It was
# added BEFORE the full experiment and is appended last, so the original five
# tests keep their priority and their counts are unchanged by the amendment.
FAILURE_LABELS = [
    "collision_from_behind",
    "freezing",
    "oscillation_near_goal",
    "corridor_deadlock",
    "timeout",
    "frontal_static_collision",
    "residual",
]

# --------------------------------------------------------------------------- #
# PPO hyperparameters (paper Table III)
# --------------------------------------------------------------------------- #
PPO_KW = dict(
    learning_rate=3e-4,
    n_steps=256,            # per env; x 8 envs = 2048 rollout length
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)
N_ENVS = 8

# --------------------------------------------------------------------------- #
# Run modes. "smoke" checks the install in ~2 min, "quick" is for a live demo
# (~15 min), "full" is the paper-sized experiment (~3.5 h on a 20-thread laptop).
# --------------------------------------------------------------------------- #
MODES = {
    "smoke": dict(
        train_steps=10_000,
        seeds_a=[0],
        wc_sweep=[2.0, 8.0, 32.0],
        seeds_b=[0],
        seeds_baseline=[0],
        eps_per_cell=2,
        eps_study_b=4,
        n_study_b_cells=1,
        n_boot=50,
    ),
    "quick": dict(
        train_steps=400_000,
        seeds_a=[0, 1],
        wc_sweep=[2.0, 8.0, 32.0, 128.0],
        seeds_b=[0],
        seeds_baseline=[0],
        eps_per_cell=30,
        eps_study_b=60,
        n_study_b_cells=2,
        n_boot=300,
    ),
    "full": dict(
        train_steps=2_000_000,
        seeds_a=[0, 1, 2, 3, 4],
        wc_sweep=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        seeds_b=[0, 1, 2],
        seeds_baseline=[0, 1, 2],
        eps_per_cell=200,
        eps_study_b=200,
        n_study_b_cells=3,
        n_boot=1000,
    ),
}


def run_name(w_c=W_C_DEFAULT, seed=0, adaptive=False):
    if adaptive:
        return f"ppo_adaptive_wc{w_c:g}_s{seed}"
    return f"ppo_wc{w_c:g}_s{seed}"
