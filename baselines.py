"""Baselines (paper Table IV), evaluated on the full grid with the same metrics:

    A* classical planner         geometric, replans on current obstacles (deterministic)
    PPO fixed reward             = the Study A policies (w_c default) -- reused from study_a.csv
    PPO fixed safety reward      w_c elevated (W_C_SAFE) and held constant
    PPO adaptive risk weighting  w_c scaled online by clearance m_t during training

Output: results/<mode>/baselines.csv

    python baselines.py --mode quick
"""

import argparse
import os
import sys

import pandas as pd

from navsim import config as C
from navsim.evaluate import run_jobs
from navsim.paths import csv_path, model_path


def main(mode="quick", workers=None):
    m = C.MODES[mode]
    n = m["eps_per_cell"]
    jobs = [dict(policy=("astar", None), cell=cell, n_episodes=n,
                 meta=dict(study="baseline", config="A* planner", run="astar", seed=0, w_c=float("nan")))
            for cell in C.ALL_CELLS]
    for s in m["seeds_baseline"]:
        for label, wc, adaptive in [("PPO fixed safety reward", C.W_C_SAFE, False),
                                    ("PPO adaptive risk", C.W_C_DEFAULT, True)]:
            name = C.run_name(wc, s, adaptive)
            path = model_path(mode, name)
            if not os.path.exists(path):
                sys.exit(f"[baselines] missing model {path}\n  -> run:  python train.py --mode {mode}")
            for cell in C.ALL_CELLS:
                jobs.append(dict(policy=("ppo", path), cell=cell, n_episodes=n,
                                 meta=dict(study="baseline", config=label, run=name, seed=s, w_c=wc)))
    # A* jobs are the slowest; submit them first so they don't straggle at the end
    df = pd.DataFrame(run_jobs(jobs, workers, desc="Baselines"))
    out = csv_path(mode, "baselines")
    df.to_csv(out, index=False)
    print(f"[baselines] wrote {len(df)} episodes -> {out}")
    print(df.groupby("config")[["success", "collision", "freeze_any"]].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="quick", choices=list(C.MODES))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.workers)
