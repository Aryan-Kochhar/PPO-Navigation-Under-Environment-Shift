"""Where everything is written. Each run mode gets its own folder so a quick demo
run never overwrites the full-size results:

    results/<mode>/models/*.zip        trained PPO policies
    results/<mode>/logs/<run>/         training curves (progress.csv, monitor.csv)
    results/<mode>/study_a.csv         episode-level results, one row per episode
    results/<mode>/study_b.csv
    results/<mode>/baselines.csv
    results/<mode>/figures/*.png
    results/<mode>/summary.md          auto-generated tables + headline numbers
"""

import os

from . import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mode_dir(mode):
    d = os.path.join(ROOT, "results", mode)
    os.makedirs(d, exist_ok=True)
    return d


def models_dir(mode):
    d = os.path.join(mode_dir(mode), "models")
    os.makedirs(d, exist_ok=True)
    return d


def logs_dir(mode):
    d = os.path.join(mode_dir(mode), "logs")
    os.makedirs(d, exist_ok=True)
    return d


def figures_dir(mode):
    d = os.path.join(mode_dir(mode), "figures")
    os.makedirs(d, exist_ok=True)
    return d


def model_path(mode, name):
    return os.path.join(models_dir(mode), name + ".zip")


def csv_path(mode, which):
    return os.path.join(mode_dir(mode), f"{which}.csv")


def required_runs(mode):
    """Every PPO training run a mode needs, de-duplicated (Study A, B and baselines
    share runs where the settings coincide, e.g. w_c = 8 seed 0)."""
    m = C.MODES[mode]
    runs = []
    for s in m["seeds_a"]:
        runs.append(dict(w_c=C.W_C_DEFAULT, seed=s))
    for wc in m["wc_sweep"]:
        for s in m["seeds_b"]:
            runs.append(dict(w_c=wc, seed=s))
    for s in m["seeds_baseline"]:
        runs.append(dict(w_c=C.W_C_SAFE, seed=s))
        runs.append(dict(w_c=C.W_C_DEFAULT, seed=s, adaptive=True))
    seen, out = set(), []
    for r in runs:
        key = C.run_name(r["w_c"], r["seed"], r.get("adaptive", False))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
