"""PPO training (Stable-Baselines3). One run = one (w_c, seed, adaptive) setting,
trained only on the centre cell of the shift grid."""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import config as C
from .env import NavEnv
from .paths import logs_dir, model_path


def train_one(mode, w_c=C.W_C_DEFAULT, seed=0, adaptive=False, steps=None, force=False, verbose=0):
    """Train one PPO policy at the centre cell. Skips if the model already exists."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    torch.set_num_threads(1)
    steps = steps or C.MODES[mode]["train_steps"]
    name = C.run_name(w_c, seed, adaptive)
    out = model_path(mode, name)
    if os.path.exists(out) and not force:
        return name, "exists", 0.0
    log_dir = os.path.join(logs_dir(mode), name)
    os.makedirs(log_dir, exist_ok=True)

    def make(i):
        def _f():
            env = NavEnv(cell=C.CENTRE, w_c=w_c, adaptive_risk=adaptive)
            env.reset(seed=seed * 1000 + i)
            return env
        return _f

    venv = VecMonitor(DummyVecEnv([make(i) for i in range(C.N_ENVS)]),
                      filename=os.path.join(log_dir, "monitor.csv"))
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **C.PPO_KW)
    model.set_logger(configure(log_dir, ["csv", "stdout"] if verbose else ["csv"]))
    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=bool(verbose))
    model.save(out)
    venv.close()
    return name, "trained", time.time() - t0


def _train_kw(kw):
    return train_one(**kw)


def train_many(mode, runs, jobs=4):
    """Train several runs in parallel processes. ``runs`` = list of kwargs dicts."""
    todo = [r for r in runs
            if not os.path.exists(model_path(mode, C.run_name(r["w_c"], r["seed"], r.get("adaptive", False))))]
    if len(runs) - len(todo):
        print(f"[train] {len(runs) - len(todo)} model(s) already exist, skipping them")
    if not todo:
        return
    jobs = max(1, min(jobs, len(todo)))
    steps = C.MODES[mode]["train_steps"]
    print(f"[train] mode={mode}: training {len(todo)} run(s) x {steps:,} steps, {jobs} in parallel")
    t0 = time.time()
    if jobs == 1:
        for r in todo:
            name, status, dt = train_one(mode, verbose=1, **r)
            print(f"[train]   {name}: {status} in {dt/60:.1f} min")
        return
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_train_kw, dict(mode=mode, **r)) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            name, status, dt = f.result()
            print(f"[train]   ({i}/{len(todo)}) {name}: {status} in {dt/60:.1f} min "
                  f"(elapsed {(time.time() - t0)/60:.1f} min)")
