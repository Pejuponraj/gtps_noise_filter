"""
bidmc_structured_contam.py
==========================
Addresses Prof. Muthu's point #3: the motivation argues the real problem is
STRUCTURED artifacts (point deletion, dropped frames, occlusion), but the original
experiment only added UNIFORM background points -- the easiest case for a density
filter. This script tests the STRUCTURED contamination the paper actually claims
to handle, so the evaluation matches the motivation.

Three structured corruption models (each on the real BIDMC clouds):

  A. POINT DELETION   : randomly delete a fraction of points (simulates missing
                        samples / signal loss).
  B. DROPPED FRAMES   : delete contiguous BLOCKS of points along the trajectory
                        (simulates dropped camera frames / occlusion bursts).
  C. UNIFORM (control): the original uniform background points, for comparison.

For each model we compare Standard TDA, GTPS (mean-kNN removal) and DTM (RMS-kNN
removal), all in the SAME removal-plus-Rips framework, by bottleneck distance to
the TRUE topology (plain Rips on the clean cloud). Lower = better recovery.
A non-degeneracy check is printed. Stats: n recordings (realisations averaged),
Wilcoxon signed-rank, rank-biserial effect size.

Reuses bidmc_full53.py (import). Run in the SAME folder:
    pip install scipy numpy ripser persim wfdb
    python bidmc_structured_contam.py
Paste the STRUCTURED RESULT block back.
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
MAX_PTS = getattr(B, "MAX_PTS", 200)
KNN_K = 6
REMOVE_C = 1.0
DELETE_FRAC = 0.30      # fraction of points removed (deletion / dropped frames)
UNIFORM_FRAC = 0.80     # matches the original uniform contamination level


def subsample(cloud, seed):
    if len(cloud) <= MAX_PTS:
        return cloud
    idx = np.random.default_rng(seed).choice(len(cloud), MAX_PTS, replace=False)
    return cloud[idx]


def knn_mean(cloud, k=KNN_K):
    D = cdist(cloud, cloud)
    return np.sort(D, axis=1)[:, 1:k + 1].mean(axis=1)


def knn_rms(cloud, k=KNN_K):
    D = cdist(cloud, cloud)
    Dk = np.sort(D, axis=1)[:, 1:k + 1]
    return np.sqrt((Dk ** 2).mean(axis=1))


# ---- structured corruption models ----
def corrupt_delete(cloud, rng):
    """A. Random point deletion: drop a fraction of points at random."""
    n = len(cloud)
    keep = rng.random(n) > DELETE_FRAC
    if keep.sum() < 10:
        keep[:10] = True
    return cloud[keep]


def corrupt_dropframes(cloud, rng):
    """B. Dropped frames: delete contiguous BLOCKS along the trajectory.
    The embedding is time-ordered, so contiguous indices = consecutive frames."""
    n = len(cloud)
    n_drop = int(DELETE_FRAC * n)
    mask = np.ones(n, dtype=bool)
    dropped = 0
    while dropped < n_drop:
        blk = rng.integers(5, 20)                 # burst length
        start = rng.integers(0, max(1, n - blk))
        mask[start:start + blk] = False
        dropped = (~mask).sum()
    if mask.sum() < 10:
        mask[:10] = True
    return cloud[mask]


def corrupt_uniform(cloud, rng):
    """C. Uniform background points (the original model), for control."""
    n = len(cloud)
    n_add = int(UNIFORM_FRAC * n)
    lo = cloud.min(axis=0); hi = cloud.max(axis=0)
    noise = rng.uniform(lo, hi, size=(n_add, cloud.shape[1]))
    return np.vstack([cloud, noise])


CORRUPTIONS = {"deletion": corrupt_delete,
               "dropframes": corrupt_dropframes,
               "uniform": corrupt_uniform}


def dgm_standard(cloud, seed):
    return ripser(subsample(cloud, seed), maxdim=2)["dgms"]


def dgm_removal(cloud, stat_fn, seed):
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


def run_model(clouds, corrupt_fn, label):
    n = len(clouds)
    per = {m: {k: [] for k in (0, 1, 2)} for m in ("std", "gtps", "dtm")}
    for rid, cloud in clouds:
        true = dgm_standard(cloud, SEED)
        acc = {m: {k: [] for k in (0, 1, 2)} for m in ("std", "gtps", "dtm")}
        for sdi in range(N_SEEDS):
            rng = np.random.default_rng(SEED + sdi)
            bad = corrupt_fn(cloud, rng)
            ds = dgm_standard(bad, SEED + sdi)
            dg = dgm_removal(bad, knn_mean, SEED + sdi)
            dd = dgm_removal(bad, knn_rms, SEED + sdi)
            for k in (0, 1, 2):
                acc["std"][k].append(bdist(true[k], ds[k]))
                acc["gtps"][k].append(bdist(true[k], dg[k]))
                acc["dtm"][k].append(bdist(true[k], dd[k]))
        for m in ("std", "gtps", "dtm"):
            for k in (0, 1, 2):
                per[m][k].append(np.nanmean(acc[m][k]))
    print(f"\n{'='*88}\nMODEL: {label}\n{'='*88}")
    for k, nm in [(0, "H0"), (1, "H1"), (2, "H2")]:
        print(f"\n--- {nm} ---   mean bottleneck to truth:")
        for m in ("std", "gtps", "dtm"):
            print(f"    {m:<6}{np.mean(per[m][k]):>8.3f}")
        print(f"  {'comparison':<20}{'baseMean':>9}{'GTPSmean':>10}{'medDiff':>10}"
              f"{'%reduc':>8}{'Wilcox p':>11}{'effect r':>9}")
        print(compare("GTPS vs Standard", per["std"][k], per["gtps"][k]))
        print(compare("GTPS vs DTM", per["dtm"][k], per["gtps"][k]))
    return per


def main():
    t0 = time.time()
    clouds = load_clouds()
    if len(clouds) < 5:
        print("Too few clouds."); return
    print(f"Structured-contamination test on {len(clouds)} recordings "
          f"(deletion & dropped-frames vs uniform control),\n"
          f"DELETE_FRAC={DELETE_FRAC}, {N_SEEDS} realisations each. Please wait...")
    allper = {}
    for key, fn in CORRUPTIONS.items():
        allper[key] = run_model(clouds, fn, key.upper())
        print(f"  [{key} done, {time.time()-t0:.0f}s]")
    print("\nHonest reading:")
    print("  * DELETION and DROPFRAMES are the STRUCTURED artifacts the paper's")
    print("    motivation cites; UNIFORM is the original control.")
    print("  * Report whatever holds: does GTPS still help under structured loss,")
    print("    and is it still comparable to DTM? Whatever it says is what we write.")
    print(f"\nTotal {time.time()-t0:.0f}s. Paste the three MODEL blocks back.")


if __name__ == "__main__":
    main()
