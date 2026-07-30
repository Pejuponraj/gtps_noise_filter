"""
bidmc_gtps_vs_dtm_fair.py
=========================
FAIR comparison of GTPS against the DTM density criterion, addressing
Prof. Muthu's "compare against an established robust method" point -- honestly.

WHY THIS IS FAIR (unlike the earlier attempt):
  * Earlier attempt used a self-stability metric (bottleneck of each method to its
    OWN clean diagram). That is GAMEABLE: a method that collapses the diagram to
    nothing scores a perfect (but meaningless) zero. We discovered this and threw
    it away.
  * Here we use the SAME ground-truth metric as the paper's main result:
    bottleneck of the method's diagram (on contaminated data) to the TRUE topology
    = a plain Vietoris-Rips diagram on the CLEAN cloud. Lower = better recovery.
    This cannot be gamed -- a degenerate diagram is FAR from the truth, not close.
  * Both GTPS and DTM are applied in the SAME framework: use the density statistic
    to REMOVE outlier points, then plain Rips. The ONLY difference is the statistic:
        GTPS : rho(x)  = MEAN of the k nearest-neighbour distances
        DTM  : delta(x)= ROOT-MEAN-SQUARE of the k nearest-neighbour distances
    Same removal rule (drop points with statistic > mean + c*std), same Rips, same
    metric. So the comparison isolates exactly the density criterion.

We also PRINT a non-degeneracy check: the clean-cloud H1 persistence for each
method must stay close to the true value (a collapsed method is disqualified).

Statistics: n = 53 recordings (5 realisations averaged within recording),
Wilcoxon signed-rank, matched-pairs rank-biserial effect size.

Reuses bidmc_full53.py exactly (import). Run in the SAME folder:
    pip install scipy numpy ripser persim wfdb
    python bidmc_gtps_vs_dtm_fair.py
Whatever the outcome -- GTPS better, comparable, or DTM better -- is reported.
Paste the FAIR RESULT block back.
"""

from __future__ import annotations
import csv
import time
import numpy as np

try:
    from scipy.stats import wilcoxon
    from scipy.spatial.distance import cdist
except Exception:
    raise SystemExit("scipy needed -> pip install scipy")
try:
    from ripser import ripser
    from persim import bottleneck as persim_bottleneck
except Exception:
    raise SystemExit("ripser/persim needed -> pip install ripser persim")

try:
    import bidmc_full53 as B
except Exception as e:
    raise SystemExit(f"import bidmc_full53.py failed (run in same folder): {e}")

SEED = B.SEED
FS_WORK = B.FS_WORK
SEG = B.SEG_SECONDS
N_SEEDS = B.N_PHASE2_SEEDS
CONTAM = B.CONTAM
MAX_PTS = getattr(B, "MAX_PTS", 200)
KNN_K = 6
REMOVE_C = 1.0


def subsample(cloud, seed):
    if len(cloud) <= MAX_PTS:
        return cloud
    idx = np.random.default_rng(seed).choice(len(cloud), MAX_PTS, replace=False)
    return cloud[idx]


def knn_mean(cloud, k=KNN_K):
    """GTPS statistic: MEAN distance to the k nearest neighbours."""
    D = cdist(cloud, cloud)
    return np.sort(D, axis=1)[:, 1:k + 1].mean(axis=1)


def knn_rms(cloud, k=KNN_K):
    """DTM statistic: RMS distance to the k nearest neighbours."""
    D = cdist(cloud, cloud)
    Dk = np.sort(D, axis=1)[:, 1:k + 1]
    return np.sqrt((Dk ** 2).mean(axis=1))


def dgm_standard(cloud, seed):
    return ripser(subsample(cloud, seed), maxdim=2)["dgms"]


def dgm_removal(cloud, stat_fn, seed):
    """Remove points whose density statistic exceeds mean + c*std, then plain Rips.
    Identical framework for GTPS and DTM; only stat_fn differs."""
    s = stat_fn(cloud)
    keep = s <= s.mean() + REMOVE_C * s.std()
    c = cloud[keep] if keep.sum() >= 10 else cloud
    return ripser(subsample(c, seed), maxdim=2)["dgms"]


def bdist(a, b):
    a = a[np.isfinite(a[:, 1])] if len(a) else np.empty((0, 2))
    b = b[np.isfinite(b[:, 1])] if len(b) else np.empty((0, 2))
    try:
        return float(persim_bottleneck(a, b))
    except Exception:
        return float("nan")


def h1max(dg):
    d = dg[1]
    d = d[np.isfinite(d[:, 1])] if len(d) else d
    return float((d[:, 1] - d[:, 0]).max()) if len(d) else 0.0


def load_clouds():
    ids = B.all_record_ids()
    clouds = []
    print(f"Loading {len(ids)} BIDMC records...")
    for i, rid in enumerate(ids, 1):
        d, status = B.load_record(rid)
        if d is None:
            continue
        ppg = B.clean_and_resample(d["ppg"], d["fs_wave"], FS_WORK, SEG)
        resp = B.clean_and_resample(d["resp"], d["fs_wave"], FS_WORK, SEG)
        spo2 = B.clean_and_resample(d["spo2"], d["fs_num"], FS_WORK, SEG)
        if ppg is None:
            continue
        ppg = B.bandpass(ppg, FS_WORK, 0.5, 4.0)
        hr = B.estimate_hr(ppg, FS_WORK)
        cloud = B.stacked_embedding(ppg, resp, spo2, FS_WORK, hr)
        if cloud is not None:
            clouds.append((rid, cloud))
    print(f"Loaded {len(clouds)} clouds.\n")
    return clouds


def rank_biserial(diff):
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    Rpos = ranks[d > 0].sum(); Rneg = ranks[d < 0].sum()
    tot = Rpos + Rneg
    return float((Rpos - Rneg) / tot) if tot > 0 else 0.0


def compare(name, a_vals, b_vals):
    """a vs b (a=baseline, b=GTPS). diff>0 => GTPS closer to truth (better)."""
    a = np.array(a_vals); b = np.array(b_vals)
    diff = a - b
    try:
        _, p = wilcoxon(a, b)
    except Exception:
        p = 1.0
    med = float(np.median(diff)); base = float(np.median(a))
    pct = 100.0 * med / base if base != 0 else 0.0
    r = rank_biserial(diff)
    verdict = ("GTPS better" if med > 1e-9 else
               ("comparable" if abs(med) < 1e-9 else "baseline better"))
    return (f"{name:<20}{a.mean():>9.3f}{b.mean():>10.3f}{med:>10.4f}"
            f"{pct:>8.1f}%{p:>11.2g}{r:>9.2f}  {verdict}")


def main():
    t0 = time.time()
    clouds = load_clouds()
    n = len(clouds)
    if n < 5:
        print("Too few clouds."); return

    # non-degeneracy check on clean clouds (mean H1 persistence per method)
    hs = {"true": [], "gtps": [], "dtm": []}
    for rid, cloud in clouds:
        hs["true"].append(h1max(dgm_standard(cloud, SEED)))
        hs["gtps"].append(h1max(dgm_removal(cloud, knn_mean, SEED)))
        hs["dtm"].append(h1max(dgm_removal(cloud, knn_rms, SEED)))
    print("Non-degeneracy check (mean clean H1 persistence; all should be similar,"
          " none collapsed to ~0):")
    print(f"    TRUE (standard) : {np.mean(hs['true']):.3f}")
    print(f"    GTPS-removal    : {np.mean(hs['gtps']):.3f}")
    print(f"    DTM-removal     : {np.mean(hs['dtm']):.3f}\n")

    print(f"Ground-truth comparison on {n} recordings x {N_SEEDS} realisations. "
          f"Please wait...\n")
    per = {m: {k: [] for k in (0, 1, 2)} for m in ("std", "gtps", "dtm")}
    raw = []
    for j, (rid, cloud) in enumerate(clouds, 1):
        true = dgm_standard(cloud, SEED)                # ground truth per recording
        acc = {m: {k: [] for k in (0, 1, 2)} for m in ("std", "gtps", "dtm")}
        for sdi in range(N_SEEDS):
            rng = np.random.default_rng(SEED + sdi)
            noisy = B.contaminate(cloud, CONTAM, rng)
            ds = dgm_standard(noisy, SEED + sdi)
            dg = dgm_removal(noisy, knn_mean, SEED + sdi)
            dd = dgm_removal(noisy, knn_rms, SEED + sdi)
            for k in (0, 1, 2):
                vs = bdist(true[k], ds[k])
                vg = bdist(true[k], dg[k])
                vd = bdist(true[k], dd[k])
                acc["std"][k].append(vs); acc["gtps"][k].append(vg); acc["dtm"][k].append(vd)
                raw.append({"record": rid, "seed": sdi, "degree": k,
                            "std": vs, "gtps": vg, "dtm": vd})
        for m in ("std", "gtps", "dtm"):
            for k in (0, 1, 2):
                per[m][k].append(np.nanmean(acc[m][k]))
        if j % 5 == 0:
            print(f"  {j}/{n} ({time.time()-t0:.0f}s)")

    with open("bidmc_gtps_dtm_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["record", "seed", "degree", "std", "gtps", "dtm"])
        w.writeheader(); w.writerows(raw)

    print("\n" + "=" * 92)
    print(f"FAIR RESULT  (n={n}; realisations averaged; bottleneck to TRUE topology, "
          f"lower=better)")
    print("=" * 92)
    for k, nm in [(0, "H0"), (1, "H1"), (2, "H2")]:
        print(f"\n--- {nm} ---   mean bottleneck to truth:")
        for m in ("std", "gtps", "dtm"):
            print(f"    {m:<6}{np.mean(per[m][k]):>8.3f}")
        print(f"  {'comparison':<20}{'baseMean':>9}{'GTPSmean':>10}{'medDiff':>10}"
              f"{'%reduc':>8}{'Wilcox p':>11}{'effect r':>9}")
        print(compare("GTPS vs Standard", per["std"][k], per["gtps"][k]))
        print(compare("GTPS vs DTM", per["dtm"][k], per["gtps"][k]))

    print("\nHonest reading:")
    print("  * 'GTPS vs DTM' isolates the density criterion (mean vs RMS kNN),")
    print("    same removal framework, same ground-truth metric.")
    print("  * medDiff>0 & r>0 => GTPS better; ~0 => comparable; <0 => DTM better.")
    print("  * Whatever it says is what we report to the reviewer.")
    print(f"\nSaved bidmc_gtps_dtm_raw.csv. Total {time.time()-t0:.0f}s. Paste the result.")


if __name__ == "__main__":
    main()
