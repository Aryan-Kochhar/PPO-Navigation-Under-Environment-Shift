"""Run the whole pipeline end to end:  train -> Study A -> Study B -> baselines -> analyze.

    python run_all.py --mode smoke    # ~2 min: checks your install, numbers are meaningless
    python run_all.py --mode quick    # ~15 min: real but small; good for the live demo
    python run_all.py --mode full     # ~3.5 hours: the paper-sized experiment

Every stage skips work that is already done (trained models are reused), so you can
stop it with Ctrl+C and re-run the same command to continue. Use --only to run
selected stages, e.g.  --only analyze   or   --only a b analyze
"""

import argparse
import os
import time

from navsim import config as C

STAGES = ["train", "a", "b", "baselines", "analyze"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="quick", choices=list(C.MODES))
    ap.add_argument("--jobs", type=int, default=None, help="parallel training processes (default: up to 4)")
    ap.add_argument("--workers", type=int, default=None, help="parallel evaluation processes (default: up to 6)")
    ap.add_argument("--only", nargs="+", choices=STAGES, default=STAGES)
    a = ap.parse_args()

    import analyze
    import baselines
    import study_a
    import study_b
    import train

    t0 = time.time()

    def banner(name):
        print(f"\n{'=' * 70}\n  {name}   (mode={a.mode}, elapsed {(time.time() - t0) / 60:.1f} min)\n{'=' * 70}")

    if "train" in a.only:
        banner("1/5  Training PPO policies at the centre cell")
        train.main(a.mode, a.jobs)
    if "a" in a.only:
        banner("2/5  Study A: failure composition across the 27-cell grid")
        study_a.main(a.mode, a.workers)
    if "b" in a.only:
        banner("3/5  Study B: safety-weight sweep / freezing-collision frontier")
        study_b.main(a.mode, a.workers)
    if "baselines" in a.only:
        banner("4/5  Baselines: A*, fixed-safety PPO, adaptive-risk PPO")
        baselines.main(a.mode, a.workers)
    if "analyze" in a.only:
        banner("5/5  Analysis: statistics, figures, summary")
        analyze.main(a.mode)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Open results/{a.mode}/summary.md and results/{a.mode}/figures/")


if __name__ == "__main__":
    main()
