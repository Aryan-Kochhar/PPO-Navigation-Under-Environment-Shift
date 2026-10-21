"""Evaluation runner shared by Study A, Study B and the baselines.

Every (policy, cell) job runs ``n_episodes`` episodes with diagnostics on and
returns one row per episode. Episode seeds depend only on (cell, episode index),
so every policy is evaluated on exactly the same scenarios (common random
numbers), which makes between-policy comparisons paired.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import config as C
from .env import NavEnv
from .metrics import EpisodeRecorder, episode_metrics
from .policies import make_policy

_POLICY_CACHE = {}


def episode_seed(cell, ep):
    return 1_000_003 * C.ALL_CELLS.index(tuple(cell)) + 7919 * ep + 12345


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    rec = EpisodeRecorder()
    done = False
    steps = 0
    while not done:
        obs, _, term, trunc, info = env.step(policy.act(env, obs))
        rec.record(env, info)
        steps += 1
        done = term or trunc
    return episode_metrics(rec, steps, env.path_length)


def eval_job(job):
    """job = dict(policy=spec, cell=(s,d,p), n_episodes=int, meta=dict)."""
    import torch
    torch.set_num_threads(1)
    spec = tuple(job["policy"])
    if spec not in _POLICY_CACHE:
        _POLICY_CACHE[spec] = make_policy(spec)
    policy = _POLICY_CACHE[spec]
    cell = tuple(job["cell"])
    env = NavEnv(cell=cell, diagnostics=True)
    rows = []
    for ep in range(job["n_episodes"]):
        row = dict(job["meta"])
        row.update(cell=C.cell_id(*cell), speed=cell[0], density=cell[1], pattern=cell[2], episode=ep)
        row.update(run_episode(env, policy, episode_seed(cell, ep)))
        rows.append(row)
    return rows


def default_workers():
    # every worker loads PyTorch (~0.5-1 GB of commit on Windows); more than ~6
    # exhausts the paging file on a 16 GB laptop ("WinError 1455")
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def run_jobs(jobs, workers=None, desc="eval"):
    """Run evaluation jobs in parallel; returns a flat list of rows."""
    from tqdm import tqdm
    workers = workers or default_workers()
    rows = []
    if workers == 1:
        for j in tqdm(jobs, desc=desc):
            rows.extend(eval_job(j))
        return rows
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(eval_job, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            rows.extend(f.result())
    return rows
