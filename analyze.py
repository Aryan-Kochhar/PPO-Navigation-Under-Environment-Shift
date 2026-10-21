"""Analysis: statistics, figures and an auto-written summary for the paper.

Reads results/<mode>/{study_a,study_b,baselines}.csv (whichever exist) and writes
    results/<mode>/figures/*.png
    results/<mode>/summary.md     tables + every headline number + RQ verdicts
    results/<mode>/summary.json   the same numbers, machine-readable

    python analyze.py --mode quick
"""

import argparse
import json
import math
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from navsim import config as C  # noqa: E402
from navsim.metrics import sens_key  # noqa: E402
from navsim.paths import csv_path, figures_dir, logs_dir, mode_dir  # noqa: E402

# --------------------------------------------------------------------------- #
# style (light chart surface, recessive chrome, fixed categorical order)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABEL_COLORS = dict(zip(C.FAILURE_LABELS[:-1], SERIES[:6]), residual=MUTED)
LABEL_NAMES = {"collision_from_behind": "Collision from behind", "freezing": "Freezing",
               "oscillation_near_goal": "Oscillation near goal", "corridor_deadlock": "Corridor deadlock",
               "timeout": "Timeout", "frontal_static_collision": "Frontal / static collision",
               "residual": "Residual (no test fires)"}
CONFIG_ORDER = ["A* planner", "PPO fixed reward", "PPO fixed safety reward", "PPO adaptive risk"]
SHORT = {"slow": "slow", "medium": "med", "fast": "fast", "sparse": "sparse", "dense": "dense",
         "linear": "lin", "random_walk": "rw", "adversarial": "adv"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def savefig(fig, mode, name):
    path = os.path.join(figures_dir(mode), name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def short_cell(cid):
    s, d, p = cid.split("-")
    return f"{SHORT[s]}/{SHORT[d]}/{SHORT[p]}"


def rate_ci(x):
    """mean and 95% Wilson-ish normal CI half width for a 0/1 array."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    p = x.mean()
    return p, 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)


# --------------------------------------------------------------------------- #
# Study A
# --------------------------------------------------------------------------- #
def cell_table(df):
    rows = []
    for (s, d, p), g in df.groupby(["speed", "density", "pattern"], sort=False):
        fails = g[g.label != "success"]
        r = dict(cell=C.cell_id(s, d, p), speed=s, density=d, pattern=p, episodes=len(g),
                 success=g.success.mean(), collision=g.collision.mean(), freezing_rate=g.freeze_any.mean(),
                 timeout=g.timeout.mean(), failures=len(fails),
                 avg_velocity=g.avg_velocity.mean(), min_obs_dist=g.min_obs_dist.mean(),
                 nav_time=g.nav_time.mean(), path_length=g.path_length.mean())
        for lab in C.FAILURE_LABELS:
            r[f"share_{lab}"] = (fails.label == lab).mean() if len(fails) else np.nan
            r[f"rate_{lab}"] = (g.label == lab).mean()
        r["residual_flag"] = bool(len(fails) and r["share_residual"] > C.RESIDUAL_LIMIT)
        rows.append(r)
    t = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(C.cell_id(*c) for c in C.ALL_CELLS)}
    return t.sort_values("cell", key=lambda s: s.map(order)).reset_index(drop=True)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def chi2_counts(tab):
    tab = np.asarray(tab, dtype=float)
    tab = tab[:, tab.sum(axis=0) > 0]
    tab = tab[tab.sum(axis=1) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(tab)
    return chi2, p


def study_a_stats(df, ct):
    out = {}
    fails = df[df.label != "success"]
    counts = pd.crosstab(fails.cell, fails.label).reindex(columns=C.FAILURE_LABELS, fill_value=0)
    chi2, p = chi2_counts(counts.values)
    out["chi2_all_cells"] = dict(chi2=chi2, p=p, dof_cells=len(counts))

    # RQ1: pairs of cells with similar success rate but different failure composition
    sr = ct.set_index("cell").success
    min_fail = max(10, int(0.1 * C.MODES[MODE]["eps_per_cell"] * df.seed.nunique()))
    pairs = []
    for a, b in combinations(counts.index, 2):
        if abs(sr[a] - sr[b]) <= 0.05 and counts.loc[a].sum() >= min_fail and counts.loc[b].sum() >= min_fail:
            _, pp = chi2_counts(counts.loc[[a, b]].values)
            if not np.isnan(pp):
                pairs.append(dict(cell_a=a, cell_b=b, sr_a=sr[a], sr_b=sr[b], p=pp))
    if pairs:
        adj = holm([q["p"] for q in pairs])
        for q, pa in zip(pairs, adj):
            q["p_holm"] = pa
        pairs.sort(key=lambda q: q["p"])
    out["matched_pairs"] = pairs
    out["n_matched_pairs"] = len(pairs)
    out["n_matched_significant"] = sum(q["p_holm"] < 0.05 for q in pairs)

    # generalisation gap, overall and per axis (other two axes at centre)
    centre_sr = sr[C.CENTRE_ID]
    out["centre_success"] = centre_sr
    out["gap_overall"] = centre_sr - sr.drop(C.CENTRE_ID).mean()
    per_seed = df.groupby(["seed", "cell"]).success.mean().unstack()
    gaps = per_seed[C.CENTRE_ID] - per_seed.drop(columns=C.CENTRE_ID).mean(axis=1)
    out["gap_overall_seed_sd"] = float(gaps.std(ddof=1)) if len(gaps) > 1 else np.nan
    axis_gap = {}
    centre_shares = ct.set_index("cell").loc[C.CENTRE_ID]
    for ai, (axis, levels) in enumerate([("speed", C.SPEEDS), ("density", C.DENSITIES), ("pattern", C.PATTERNS)]):
        for lev in (levels[0], levels[2]):
            cell = list(C.CENTRE)
            cell[ai] = lev
            cid = C.cell_id(*cell)
            row = ct.set_index("cell").loc[cid]
            axis_gap[f"{axis}={lev}"] = dict(
                cell=cid, success=row.success, gap=centre_sr - row.success,
                freezing_share=row.share_freezing, cfb_share=row.share_collision_from_behind,
                deadlock_share=row.share_corridor_deadlock, frontal_share=row.share_frontal_static_collision,
                freezing_share_centre=centre_shares.share_freezing,
                cfb_share_centre=centre_shares.share_collision_from_behind)
    out["axis_gap"] = axis_gap
    ranking = fails[fails.label != "residual"].label.value_counts()
    out["failure_ranking"] = ranking.to_dict()   # all labelled categories (Study B excludes frontal/static)
    out["residual_share_overall"] = (fails.label == "residual").mean() if len(fails) else np.nan
    out["residual_flag_cells"] = ct[ct.residual_flag].cell.tolist()
    out["residual_collision_kinds"] = fails[fails.label == "residual"].collision_kind.replace("", "not a collision").value_counts().to_dict()
    frontal = fails[fails.label == "frontal_static_collision"]
    out["frontal_static_share_overall"] = len(frontal) / len(fails) if len(fails) else np.nan
    out["frontal_static_breakdown"] = frontal.collision_kind.value_counts().to_dict()
    # the two catch-all categories (timeout, frontal/static collision) carry no
    # mechanism; a large share means the specific tests explain little
    out["catch_all_share_overall"] = fails.label.isin(["timeout", "frontal_static_collision"]).mean() if len(fails) else np.nan
    return out


def fig_a_heatmap(ct, mode):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, pat in zip(axes, C.PATTERNS):
        sub = ct[ct.pattern == pat].set_index(["speed", "density"]).success
        grid = np.array([[sub.get((s, d), np.nan) for d in C.DENSITIES] for s in C.SPEEDS])
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="lower")
        for i in range(3):
            for j in range(3):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color=SURFACE if v > 0.55 else INK, fontweight="bold" if (C.SPEEDS[i], C.DENSITIES[j], pat) == C.CENTRE else "normal")
        if pat == C.CENTRE[2]:
            ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
        ax.set_xticks(range(3), C.DENSITIES)
        ax.set_yticks(range(3), C.SPEEDS)
        ax.set_title(pat.replace("_", " "), loc="center", fontsize=10)
        ax.set_xlabel("obstacle density")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("obstacle speed")
    fig.suptitle("Study A - success rate across the 3x3x3 shift grid (outlined = training cell)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return savefig(fig, mode, "a1_success_heatmap.png")


def _stacked(ax, labels_rows, ct_rows, horizontal=True):
    left = np.zeros(len(ct_rows))
    pos = np.arange(len(ct_rows))
    for lab in C.FAILURE_LABELS:
        vals = np.array([r[f"rate_{lab}"] for r in ct_rows])
        kw = dict(color=LABEL_COLORS[lab], edgecolor=SURFACE, linewidth=1.0, label=LABEL_NAMES[lab])
        if horizontal:
            ax.barh(pos, vals, left=left, height=0.72, **kw)
        else:
            ax.bar(pos, vals, bottom=left, width=0.68, **kw)
        left += vals
    return pos, left


def fig_a_composition(ct, mode):
    rows = [r for _, r in ct.iterrows()]
    fig, ax = plt.subplots(figsize=(8.5, 8.6))
    pos, tot = _stacked(ax, None, rows)
    ax.set_yticks(pos, [short_cell(r["cell"]) + ("  (train)" if r["cell"] == C.CENTRE_ID else "") for r in rows],
                  fontsize=8.5)
    for p, t, r in zip(pos, tot, rows):
        ax.text(t + 0.01, p, f"SR {r['success']:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.0, tot.max() + 0.15))
    ax.set_xlabel("fraction of all episodes (bar length = failure rate)")
    ax.grid(axis="y", visible=False)
    for k in (9, 18):
        ax.axhline(k - 0.5, color=AXIS, lw=0.8)
    ax.set_title("Study A - failure composition per cell (speed / density / pattern)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, mode, "a2_failure_composition.png")


def fig_a_one_axis(ct, mode):
    t = ct.set_index("cell")
    cells, names = [C.CENTRE_ID], ["centre\n(train)"]
    for ai, levels in enumerate([C.SPEEDS, C.DENSITIES, C.PATTERNS]):
        for lev in (levels[0], levels[2]):
            cell = list(C.CENTRE)
            cell[ai] = lev
            cells.append(C.cell_id(*cell))
            names.append(f"{['speed', 'density', 'pattern'][ai]}\n{lev.replace('_', ' ')}")
    rows = [t.loc[c] for c in cells]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    pos, tot = _stacked(ax, None, rows, horizontal=False)
    for p, top, r in zip(pos, tot, rows):
        ax.text(p, top + 0.01, f"SR {r['success']:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(pos, names, fontsize=8.5)
    ax.set_ylabel("fraction of episodes")
    ax.grid(axis="x", visible=False)
    ax.set_title("Study A - one axis shifted at a time (other two at centre)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, mode, "a3_one_axis_shift.png")


def fig_sensitivity(df, mode):
    grid = np.array([[df[sens_key(dfr, n)].mean() for n in C.SENS_N] for dfr in C.SENS_DELTA_FRACS])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(grid, cmap=cmap, origin="lower", vmin=0, vmax=max(grid.max(), 1e-6))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            chosen = C.SENS_DELTA_FRACS[i] == 0.10 and C.SENS_N[j] == C.FREEZE_N
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if grid[i, j] > 0.6 * grid.max() else INK, fontweight="bold" if chosen else "normal")
    ax.set_xticks(range(len(C.SENS_N)), C.SENS_N)
    ax.set_yticks(range(len(C.SENS_DELTA_FRACS)), [f"{d:.2f}" for d in C.SENS_DELTA_FRACS])
    ax.set_xlabel("window N (steps)")
    ax.set_ylabel("delta (x v_max * dt)")
    ax.grid(False)
    ax.set_title("Freezing rate sensitivity to (delta, N)\nStudy A, all cells; bold = pre-registered", fontsize=10)
    fig.tight_layout()
    return savefig(fig, mode, "a4_freezing_sensitivity.png"), grid


# --------------------------------------------------------------------------- #
# Study B
# --------------------------------------------------------------------------- #
def knee_point(wcs, coll, frz):
    """Knee of the collision-freezing frontier (Kneedle: max distance between the
    normalised collision reduction and the normalised freezing increase along a
    PCHIP fit in log2(w_c)); coincides with max curvature for a convex frontier."""
    x = np.log2(np.asarray(wcs, dtype=float))
    o = np.argsort(x)
    x, c, f = x[o], np.asarray(coll, float)[o], np.asarray(frz, float)[o]
    if len(x) < 3:
        return np.nan
    xf = np.linspace(x[0], x[-1], 400)
    cf, ff = PchipInterpolator(x, c)(xf), PchipInterpolator(x, f)(xf)
    cr, fr = cf.max() - cf.min(), ff.max() - ff.min()
    if cr < 1e-9 or fr < 1e-9:
        return np.nan
    score = (cf.max() - cf) / cr - (ff - ff.min()) / fr
    return float(2 ** xf[int(np.argmax(score))])


def bootstrap_knee(g, n_boot, rng):
    """Hierarchical bootstrap: resample seeds, then episodes within seed."""
    wcs = sorted(g.w_c.unique())
    data = {wc: [(s.collision.values, s.freeze_any.values) for _, s in g[g.w_c == wc].groupby("seed")] for wc in wcs}
    knees = []
    for _ in range(n_boot):
        cs, fs = [], []
        for wc in wcs:
            seeds = data[wc]
            pick = rng.integers(0, len(seeds), size=len(seeds))
            cc, ff = [], []
            for k in pick:
                c, f = seeds[k]
                idx = rng.integers(0, len(c), size=len(c))
                cc.append(c[idx])
                ff.append(f[idx])
            cs.append(np.concatenate(cc).mean())
            fs.append(np.concatenate(ff).mean())
        knees.append(knee_point(wcs, cs, fs))
    knees = np.array(knees)
    knees = knees[~np.isnan(knees)]
    if len(knees) < max(10, 0.5 * n_boot):
        return np.nan, np.nan, len(knees) / max(n_boot, 1)
    return float(np.percentile(knees, 2.5)), float(np.percentile(knees, 97.5)), len(knees) / n_boot


def study_b_stats(b, n_boot):
    rng = np.random.default_rng(0)
    tab = b.groupby(["cell", "w_c"]).agg(collision=("collision", "mean"), freezing=("freeze_any", "mean"),
                                        success=("success", "mean"), timeout=("timeout", "mean"),
                                        episodes=("success", "size")).reset_index()
    knees = {}
    for cell, g in b.groupby("cell"):
        t = tab[tab.cell == cell].sort_values("w_c")
        k = knee_point(t.w_c, t.collision, t.freezing)
        lo, hi, frac = bootstrap_knee(g, n_boot, rng)
        mono_c = bool(np.all(np.diff(t.collision.values) <= 0.02))
        mono_f = bool(np.all(np.diff(t.freezing.values) >= -0.02))
        knees[cell] = dict(knee=k, ci_low=lo, ci_high=hi, boot_valid_frac=frac,
                           collision_monotone=mono_c, freezing_monotone=mono_f,
                           freezing_range=float(t.freezing.max() - t.freezing.min()),
                           collision_range=float(t.collision.max() - t.collision.min()))
    return tab, knees


def fig_b_frontier(tab, knees, mode):
    cells = [C.CENTRE_ID] + [c for c in tab.cell.unique() if c != C.CENTRE_ID]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for i, cell in enumerate(cells):
        t = tab[tab.cell == cell].sort_values("w_c")
        col = SERIES[i % len(SERIES)]
        ax.plot(t.collision, t.freezing, "-o", color=col, label=short_cell(cell) + (" (train)" if cell == C.CENTRE_ID else ""),
                markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=7)
        for _, r in t.iterrows():
            ax.annotate(f"{r.w_c:g}", (r.collision, r.freezing), textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5, color=INK2)
        k = knees.get(cell, {}).get("knee", np.nan)
        if not np.isnan(k):
            x = np.log2(t.w_c.values)
            kc = PchipInterpolator(x, t.collision.values)(np.log2(k))
            kf = PchipInterpolator(x, t.freezing.values)(np.log2(k))
            ax.plot(kc, kf, marker="*", markersize=16, color=col, markeredgecolor=INK, markeredgewidth=0.8, zorder=5)
    ax.set_xlabel("collision rate")
    ax.set_ylabel("freezing rate")
    ax.set_title("Study B - freezing-collision frontier (labels = w_c, star = knee)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, mode, "b1_frontier.png")


def fig_b_rates(tab, knees, mode):
    cells = [C.CENTRE_ID] + [c for c in tab.cell.unique() if c != C.CENTRE_ID]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for i, cell in enumerate(cells):
        t = tab[tab.cell == cell].sort_values("w_c")
        col = SERIES[i % len(SERIES)]
        lab = short_cell(cell) + (" (train)" if cell == C.CENTRE_ID else "")
        for ax, key in zip(axes, ["collision", "freezing", "success"]):
            ax.plot(t.w_c, t[key], "-o", color=col, label=lab, markeredgecolor=SURFACE, markeredgewidth=1.2)
        kn = knees.get(cell, {})
        # CI band for the training cell only; overlapping bands for every cell are unreadable
        if cell == C.CENTRE_ID and not np.isnan(kn.get("ci_low", np.nan)):
            for ax in axes[:2]:
                ax.axvspan(kn["ci_low"], kn["ci_high"], color=col, alpha=0.08, lw=0)
        if not np.isnan(kn.get("knee", np.nan)):
            for ax in axes[:2]:
                ax.axvline(kn["knee"], color=col, lw=1.0, ls="--")
    for ax, ttl in zip(axes, ["Collision rate", "Freezing rate", "Success rate"]):
        ax.set_xscale("log", base=2)
        wcs = sorted(tab.w_c.unique())
        ax.set_xticks(wcs, [f"{w:g}" for w in wcs])
        ax.minorticks_off()
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("safety weight w_c (log scale)")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle("Study B - rates vs safety weight (dashed = knee, band = training-cell bootstrap 95% CI)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return savefig(fig, mode, "b2_rates_vs_wc.png")


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_table(allb):
    rows = []
    for cfg, g in allb.groupby("config"):
        fails = g[g.label != "success"]
        centre = g[g.cell == C.CENTRE_ID].success.mean()
        outer = g[g.cell != C.CENTRE_ID].success.mean()
        r = dict(config=cfg, success=g.success.mean(), collision=g.collision.mean(), freezing_rate=g.freeze_any.mean(),
                 timeout=g.timeout.mean(), centre_success=centre, gap=centre - outer,
                 adversarial_success=g[g.pattern == "adversarial"].success.mean(),
                 avg_velocity=g.avg_velocity.mean(), nav_time=g.nav_time.mean())
        for lab in C.FAILURE_LABELS:
            r[f"share_{lab}"] = (fails.label == lab).mean() if len(fails) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t["o"] = t.config.map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return t.sort_values("o").drop(columns="o").reset_index(drop=True)


def fig_baselines(allb, bt, mode):
    groups = [("centre (train)", C.CENTRE)]
    for ai, levels in enumerate([C.SPEEDS, C.DENSITIES, C.PATTERNS]):
        for lev in (levels[0], levels[2]):
            cell = list(C.CENTRE)
            cell[ai] = lev
            groups.append((f"{['speed', 'density', 'pattern'][ai]}\n{lev.replace('_', ' ')}", tuple(cell)))
    cfgs = [c for c in CONFIG_ORDER if c in allb.config.unique()]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    w = 0.8 / len(cfgs)
    for i, cfg in enumerate(cfgs):
        vals = [allb[(allb.config == cfg) & (allb.cell == C.cell_id(*cell))].success.mean() for _, cell in groups]
        ax.bar(np.arange(len(groups)) + (i - (len(cfgs) - 1) / 2) * w, vals, width=w * 0.92,
               color=SERIES[i], label=cfg, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_title("Baselines - success under one-axis shift")
    ax.legend(fontsize=8.5, ncol=len(cfgs), loc="upper center", bbox_to_anchor=(0.5, -0.14))

    rows = [dict(bt.set_index("config").loc[c]) for c in cfgs]
    for r, c in zip(rows, cfgs):
        g = allb[allb.config == c]
        for lab in C.FAILURE_LABELS:
            r[f"rate_{lab}"] = (g.label == lab).mean()
    pos, tot = _stacked(ax2, None, rows)
    ax2.set_yticks(pos, cfgs, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("fraction of all episodes, all 27 cells")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Failure composition")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    fig.tight_layout()
    return savefig(fig, mode, "c1_baselines.png")


def fig_training(mode):
    d = logs_dir(mode)
    runs = sorted(r for r in os.listdir(d) if os.path.exists(os.path.join(d, r, "monitor.csv")))
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4))
    wcs = sorted({float(r.split("_wc")[1].split("_")[0]) for r in runs if "adaptive" not in r})
    for r in runs:
        try:
            m = pd.read_csv(os.path.join(d, r, "monitor.csv"), skiprows=1)
        except Exception:
            continue
        if m.empty:
            continue
        m["steps"] = m.l.cumsum()
        roll = m.r.rolling(max(5, len(m) // 40), min_periods=1).mean()
        if "adaptive" in r:
            col = SERIES[3]
        else:
            wc = float(r.split("_wc")[1].split("_")[0])
            col = SEQ[min(len(SEQ) - 1, 2 + int(round(4 * wcs.index(wc) / max(1, len(wcs) - 1))))]
        ax.plot(m.steps, roll, color=col, lw=1.3, alpha=0.9)
    handles = [plt.Line2D([], [], color=SEQ[2 + int(round(4 * i / max(1, len(wcs) - 1)))], label=f"w_c={wc:g}")
               for i, wc in enumerate(wcs)]
    if any("adaptive" in r for r in runs):
        handles.append(plt.Line2D([], [], color=SERIES[3], label="adaptive risk"))
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return (rolling mean)")
    ax.set_title("Training curves at the centre cell (all runs)")
    fig.tight_layout()
    return savefig(fig, mode, "d1_training_curves.png")


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def md_table(df, cols, names=None, nd=3):
    names = names or cols
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], nd) if not isinstance(r[c], str) else r[c] for c in cols) + " |")
    return "\n".join(lines)


def main(mode="quick"):
    global MODE
    MODE = mode
    style()
    m = C.MODES[mode]
    md = [f"# Results summary - mode `{mode}`", "",
          f"Auto-generated by `analyze.py`. Training steps per run: {m['train_steps']:,}. "
          f"Thresholds: delta={C.FREEZE_DELTA:g} m/step, N={C.FREEZE_N}, h={C.AFREE_HORIZON}, "
          f"R_g={C.OSC_RADIUS}, M={C.OSC_MIN_FLIPS}, N_o={C.OSC_WINDOW}, eps_d={C.DEADLOCK_EPS}, "
          f"lateral={C.DEADLOCK_LATERAL}.", ""]
    summary = {"mode": mode}
    figs = []

    a_path = csv_path(mode, "study_a")
    a = pd.read_csv(a_path, keep_default_na=False, na_values=[""]) if os.path.exists(a_path) else None
    if a is not None:
        a["collision_kind"] = a["collision_kind"].fillna("")
        ct = cell_table(a)
        ct.to_csv(os.path.join(mode_dir(mode), "study_a_cells.csv"), index=False)
        st = study_a_stats(a, ct)
        summary["study_a"] = st
        figs += [fig_a_heatmap(ct, mode), fig_a_composition(ct, mode), fig_a_one_axis(ct, mode)]
        f, sens = fig_sensitivity(a, mode)
        figs.append(f)
        summary["sensitivity"] = {f"d{d:.2f}_N{n}": float(sens[i, j]) for i, d in enumerate(C.SENS_DELTA_FRACS)
                                  for j, n in enumerate(C.SENS_N)}

        md += ["## Study A - failure composition under shift", "",
               f"Episodes: {len(a)} ({a.seed.nunique()} seed(s) x 27 cells x {m['eps_per_cell']}).", "",
               f"- Centre-cell success rate: **{fmt(st['centre_success'])}**",
               f"- Generalisation gap G = SR(centre) - mean SR(other 26): **{fmt(st['gap_overall'])}** "
               f"(seed sd {fmt(st['gap_overall_seed_sd'])})",
               f"- Chi-squared homogeneity of failure categories across all cells: "
               f"chi2 = {fmt(st['chi2_all_cells']['chi2'], 1)}, p = {fmt(st['chi2_all_cells']['p'], 4)}",
               f"- Failure ranking (named categories): {st['failure_ranking']}",
               f"- Frontal/static collision share of all failures (amended category): "
               f"**{fmt(st['frontal_static_share_overall'])}**; breakdown: {st['frontal_static_breakdown']}",
               f"- Catch-all share (timeout + frontal/static collision): {fmt(st['catch_all_share_overall'])}",
               f"- Residual share of all failures: **{fmt(st['residual_share_overall'])}**; "
               f"cells over the {int(C.RESIDUAL_LIMIT*100)}% limit: {len(st['residual_flag_cells'])}/27; "
               f"residual breakdown: {st['residual_collision_kinds']}", ""]
        md += ["### Per-axis gap (one axis shifted, others at centre)", "",
               "| shift | SR | gap vs centre | freezing share | collision-from-behind share | deadlock share "
               "| frontal/static share |",
               "|---|---|---|---|---|---|---|"]
        for k, v in st["axis_gap"].items():
            md.append(f"| {k} | {fmt(v['success'])} | {fmt(v['gap'])} | {fmt(v['freezing_share'])} | "
                      f"{fmt(v['cfb_share'])} | {fmt(v['deadlock_share'])} | {fmt(v['frontal_share'])} |")
        md += ["", f"Centre shares: freezing {fmt(st['axis_gap']['speed=slow']['freezing_share_centre'])}, "
               f"collision-from-behind {fmt(st['axis_gap']['speed=slow']['cfb_share_centre'])}.", ""]
        md += ["### RQ1 - matched success, different failures", "",
               f"Cell pairs with |dSR| <= 0.05 and enough failures: {st['n_matched_pairs']}; "
               f"significant after Holm correction (p<0.05): **{st['n_matched_significant']}**.", ""]
        if st["matched_pairs"]:
            md += ["| cell A | cell B | SR A | SR B | p | p (Holm) |", "|---|---|---|---|---|---|"]
            for q in st["matched_pairs"][:10]:
                md.append(f"| {q['cell_a']} | {q['cell_b']} | {fmt(q['sr_a'])} | {fmt(q['sr_b'])} | "
                          f"{fmt(q['p'], 4)} | {fmt(q['p_holm'], 4)} |")
            md.append("")
        md += ["### Per-cell table", "",
               md_table(ct, ["cell", "success", "collision", "freezing_rate", "timeout",
                             "share_collision_from_behind", "share_freezing", "share_oscillation_near_goal",
                             "share_corridor_deadlock", "share_timeout", "share_frontal_static_collision",
                             "share_residual"],
                        ["cell", "SR", "coll", "freeze", "timeout", "%cfb", "%freeze", "%osc", "%deadlock",
                         "%timeout", "%frontal/static", "%residual"]), ""]
    else:
        md += ["## Study A", "", "_study_a.csv not found - run `python study_a.py`._", ""]

    b_path = csv_path(mode, "study_b")
    if os.path.exists(b_path):
        b = pd.read_csv(b_path)
        tab, knees = study_b_stats(b, m["n_boot"])
        tab.to_csv(os.path.join(mode_dir(mode), "study_b_rates.csv"), index=False)
        summary["study_b"] = dict(knees=knees)
        figs += [fig_b_frontier(tab, knees, mode), fig_b_rates(tab, knees, mode)]
        md += ["## Study B - freezing-collision frontier", "",
               "| cell | knee w_c* | 95% CI | collision monotone down | freezing monotone up | freezing range |",
               "|---|---|---|---|---|---|"]
        for cell, k in knees.items():
            md.append(f"| {cell} | {fmt(k['knee'], 2)} | [{fmt(k['ci_low'], 2)}, {fmt(k['ci_high'], 2)}] | "
                      f"{k['collision_monotone']} | {k['freezing_monotone']} | {fmt(k['freezing_range'])} |")
        md += ["", md_table(tab, ["cell", "w_c", "collision", "freezing", "success", "timeout"]), ""]
    else:
        md += ["## Study B", "", "_study_b.csv not found - run `python study_b.py`._", ""]

    c_path = csv_path(mode, "baselines")
    if os.path.exists(c_path):
        c = pd.read_csv(c_path, keep_default_na=False, na_values=[""])
        parts = [c]
        if a is not None:
            parts.append(a.assign(config="PPO fixed reward"))
        allb = pd.concat(parts, ignore_index=True)
        bt = baseline_table(allb)
        summary["baselines"] = bt.to_dict(orient="records")
        figs.append(fig_baselines(allb, bt, mode))
        md += ["## Baselines", "",
               md_table(bt, ["config", "success", "collision", "freezing_rate", "timeout", "centre_success", "gap",
                             "adversarial_success", "share_frontal_static_collision", "share_residual"],
                        ["config", "SR", "coll", "freeze", "timeout", "SR centre", "gap G", "SR adversarial",
                         "%frontal/static", "%residual"]), ""]
    else:
        md += ["## Baselines", "", "_baselines.csv not found - run `python baselines.py`._", ""]

    tf = fig_training(mode)
    if tf:
        figs.append(tf)

    md += ["## Pre-registered questions - verdicts", ""] + verdicts(summary) + ["", "## Figures", ""]
    md += [f"- `{os.path.relpath(f, mode_dir(mode))}`" for f in figs if f]
    with open(os.path.join(mode_dir(mode), "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(mode_dir(mode), "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else
                  (o.item() if hasattr(o, "item") else str(o)))
    print(f"[analyze] wrote {len([f for f in figs if f])} figures and summary -> {mode_dir(mode)}")
    print("\n".join(verdicts(summary)))


def verdicts(s):
    out = []
    a = s.get("study_a")
    if a:
        sig = a["n_matched_significant"]
        out.append(f"- **RQ1** (composition changes where SR does not): "
                   + (f"SUPPORTED - {sig} matched-SR cell pair(s) have significantly different failure "
                      f"distributions after Holm correction." if sig else
                      f"NOT SUPPORTED at this sample size - 0 of {a['n_matched_pairs']} matched pairs differ "
                      f"significantly (overall chi2 p = {fmt(a['chi2_all_cells']['p'], 4)})."))
        ag = a["axis_gap"]
        worst = max(ag, key=lambda k: ag[k]["gap"])
        dens_frz = ag["density=dense"]["freezing_share"] > ag["density=dense"]["freezing_share_centre"]
        speed_cfb = ag["speed=fast"]["cfb_share"] > ag["speed=fast"]["cfb_share_centre"]
        out.append(f"- **RQ2** (attribution to a shift axis): largest one-axis drop is `{worst}` "
                   f"(gap {fmt(ag[worst]['gap'])}). Dense -> more freezing share: {dens_frz}. "
                   f"Fast -> more collision-from-behind share: {speed_cfb}.")
        res = a["residual_share_overall"]
        out.append(f"- **Taxonomy coverage**: residual = {fmt(res)} of failures; "
                   + ("pre-registration UNDER-COVERS the failure space in "
                      f"{len(a['residual_flag_cells'])} cell(s) (report as a negative result; "
                      f"breakdown {a['residual_collision_kinds']})." if a["residual_flag_cells"]
                      else "within the 10% limit in every cell.")
                   + f" Note: since the amendment, every collision and every timeout has a label, so the residual "
                     f"is near-empty by construction; the informative coverage figure is the catch-all share "
                     f"(timeout + frontal/static) = {fmt(a['catch_all_share_overall'])}.")
    b = s.get("study_b")
    if b:
        k = b["knees"]
        centre = k.get(C.CENTRE_ID, {})
        out.append(f"- **RQ3** (knee in the frontier): centre-cell knee w_c* = {fmt(centre.get('knee'), 2)} "
                   f"[95% CI {fmt(centre.get('ci_low'), 2)}, {fmt(centre.get('ci_high'), 2)}]"
                   + ("" if not np.isnan(centre.get("knee", np.nan)) else
                      " - no knee: one of the rates does not vary across the sweep (see alternatives in Sec. VII-B)."))
        shifted = {c: v for c, v in k.items() if c != C.CENTRE_ID}
        moved = []
        for c, v in shifted.items():
            if any(np.isnan([v["knee"], centre.get("ci_low", np.nan), v.get("ci_low", np.nan)])):
                moved.append(f"{c}: undetermined")
            else:
                overlap = not (v["ci_high"] < centre["ci_low"] or v["ci_low"] > centre["ci_high"])
                direction = "lower" if v["knee"] < centre["knee"] else "higher"
                moved.append(f"{c}: knee {fmt(v['knee'], 2)} ({direction}; CIs {'overlap' if overlap else 'DO NOT overlap'})")
        out.append("- **RQ4** (knee moves under shift): " + ("; ".join(moved) if moved else "no shifted cells evaluated"))
    return out


MODE = "quick"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="quick", choices=list(C.MODES))
    main(ap.parse_args().mode)
