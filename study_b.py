"""Study B: the freezing-collision frontier.

Depends on Study A. Reads study_a.csv, ranks the failure categories, and picks the
shift cells Study A flags as most freezing-prone. Then evaluates every policy of
the safety-weight sweep (w_c in the mode's sweep x seeds) on the centre cell plus
those flagged cells. Output: results/<mode>/study_b.csv and study_b_cells.json.

    python study_b.py --mode quick
"""

import argparse
import json
import os
import sys

import pandas as pd

from navsim import config as C
from navsim.evaluate import run_jobs
from navsim.paths import csv_path, mode_dir, model_path


def select_cells(mode):
    """Use Study A's output to choose Study B's evaluation cells."""
    m = C.MODES[mode]
    a_path = csv_path(mode, "study_a")
    if not os.path.exists(a_path):
        sys.exit(f"[study B] Study B depends on Study A. Missing {a_path}\n  -> run:  python study_a.py --mode {mode}")
    a = pd.read_csv(a_path)
    fails = a[a.label != "success"]
    # the amended frontal/static category names no mechanism and is excluded from
    # the ranking, so the amendment cannot change which mode Study B targets
    ranking = fails[~fails.label.isin(["residual", "frontal_static_collision"])].label.value_counts()
    top = ranking.index[0] if len(ranking) else "freezing"
    if top != "freezing":
        print(f"[study B] NOTE: Study A ranks '{top}' as the most frequent named failure, not freezing. "
              "The sweep variable (w_c) is unchanged; report this with the results.")
    per_cell = a[a.cell != C.CENTRE_ID].groupby(["speed", "density", "pattern"]).freeze_any.mean()
    flagged = list(per_cell.sort_values(ascending=False).index[: m["n_study_b_cells"]])
    cells = [C.CENTRE] + [tuple(c) for c in flagged]
    info = dict(top_failure_category=top, failure_ranking=ranking.to_dict(),
                cells=[C.cell_id(*c) for c in cells],
                freezing_rate_in_study_a={C.cell_id(*c): float(per_cell[c]) for c in flagged})
    with open(os.path.join(mode_dir(mode), "study_b_cells.json"), "w") as f:
        json.dump(info, f, indent=2)
    return cells, info


def main(mode="quick", workers=None):
    m = C.MODES[mode]
    cells, info = select_cells(mode)
    print(f"[study B] evaluation cells (centre + most freezing-prone in Study A): {info['cells']}")
    jobs = []
    for wc in m["wc_sweep"]:
        for s in m["seeds_b"]:
            name = C.run_name(wc, s)
            path = model_path(mode, name)
            if not os.path.exists(path):
                sys.exit(f"[study B] missing model {path}\n  -> run:  python train.py --mode {mode}")
            for cell in cells:
                jobs.append(dict(policy=("ppo", path), cell=cell, n_episodes=m["eps_study_b"],
                                 meta=dict(study="B", config=f"PPO w_c={wc:g}", run=name, seed=s, w_c=wc)))
    print(f"[study B] {len(m['wc_sweep'])} weights x {len(m['seeds_b'])} seed(s) x {len(cells)} cells "
          f"x {m['eps_study_b']} episodes")
    df = pd.DataFrame(run_jobs(jobs, workers, desc="Study B"))
    out = csv_path(mode, "study_b")
    df.to_csv(out, index=False)
    print(f"[study B] wrote {len(df)} episodes -> {out}")
    print(df.groupby("w_c")[["collision", "freeze_any", "success"]].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="quick", choices=list(C.MODES))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.workers)
