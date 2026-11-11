"""Train PPO policies at the centre cell.

Examples
    python train.py --mode quick                 # every run the quick mode needs
    python train.py --mode full --jobs 8         # every run at full size, 8 in parallel
    python train.py --mode quick --wc 8 --seed 0 # a single run (shows a progress bar)
"""

import argparse
import os

from navsim import config as C
from navsim.paths import required_runs
from navsim.training import train_many, train_one


def main(mode="quick", jobs=None):
    jobs = jobs or max(1, min(4, (os.cpu_count() or 2) // 2))   # each job holds PyTorch in memory
    train_many(mode, required_runs(mode), jobs=jobs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="quick", choices=list(C.MODES))
    ap.add_argument("--jobs", type=int, default=None, help="parallel training processes")
    ap.add_argument("--wc", type=float, default=None, help="train only this safety weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adaptive", action="store_true", help="adaptive risk weighting")
    ap.add_argument("--steps", type=int, default=None, help="override training steps")
    ap.add_argument("--force", action="store_true", help="retrain even if the model exists")
    a = ap.parse_args()
    if a.wc is None and not a.adaptive:
        main(a.mode, a.jobs)
    else:
        wc = a.wc if a.wc is not None else C.W_C_DEFAULT
        name, status, dt = train_one(a.mode, w_c=wc, seed=a.seed, adaptive=a.adaptive,
                                     steps=a.steps, force=a.force, verbose=1)
        print(f"{name}: {status} ({dt/60:.1f} min)")
