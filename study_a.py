"""Study A: failure composition under shift.

The PPO policy trained at the centre cell (w_c = default) is frozen and evaluated
on all 27 cells of the shift grid, for every Study A seed. Every failed episode
gets one taxonomy label. Output: results/<mode>/study_a.csv (one row per episode).

    python study_a.py --mode quick
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
    jobs = []
    for s in m["seeds_a"]:
        name = C.run_name(C.W_C_DEFAULT, s)
        path = model_path(mode, name)
        if not os.path.exists(path):
            sys.exit(f"[study A] missing model {path}\n  -> run:  python train.py --mode {mode}")
        for cell in C.ALL_CELLS:
            jobs.append(dict(policy=("ppo", path), cell=cell, n_episodes=m["eps_per_cell"],
                             meta=dict(study="A", config="PPO fixed reward", run=name, seed=s, w_c=C.W_C_DEFAULT)))
    print(f"[study A] {len(m['seeds_a'])} seed(s) x 27 cells x {m['eps_per_cell']} episodes")
    df = pd.DataFrame(run_jobs(jobs, workers, desc="Study A"))
    out = csv_path(mode, "study_a")
    df.to_csv(out, index=False)

    centre = df[df.cell == C.CENTRE_ID]
    print(f"[study A] wrote {len(df)} episodes -> {out}")
    print(f"[study A] centre cell success rate {centre.success.mean():.2f}, overall {df.success.mean():.2f}")
    print(df[df.label != "success"].label.value_counts().to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="quick", choices=list(C.MODES))
    ap.add_argument("--workers", type=int, default=None, help="parallel evaluation processes")
    a = ap.parse_args()
    main(a.mode, a.workers)
