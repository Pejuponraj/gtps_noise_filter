"""
bidmc_stats_fixed.py
====================
Fixes the pseudoreplication flagged in review. The original phase-2 pooled all
53*5 = 265 (recording x noise-realisation) bottleneck values and ran a paired
t-test as if they were independent -- but the 5 realisations within a recording
are correlated, which inflates the p-values.

This script recomputes the SAME phase-2 comparison but with correct statistics:

  1. n = 53:  the 5 contamination realisations of each recording are AVERAGED,
     giving one (Std, GTPS) pair per recording -> 53 independent paired samples.
  2. Wilcoxon signed-rank test (not paired t-test), because bottleneck distances
     are non-negative and skewed.
  3. Effect sizes reported explicitly: median paired difference, percent
     reduction, and matched-pairs rank-biserial correlation -- so a small effect
     is shown as small rather than hidden behind a tiny p-value.
  4. The raw 53x5x3 bottleneck matrix is saved to bidmc_bottleneck_raw.csv, so any
     future statistic can be recomputed WITHOUT re-running the expensive homology.

It reuses the exact functions/constants of bidmc_full53.py (import), so the
numbers are directly comparable. Run it in the SAME folder as bidmc_full53.py:

    python bidmc_stats_fixed.py

Expect roughly the same run time as the original phase-2 (~30-60 min; the earlier
9-hour figure was the machine sleeping). Paste the CORRECTED STATS block back.
"""

from __future__ import annotations
import csv
import time
import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    raise SystemExit("scipy needed -> pip install scipy")

# reuse the exact pipeline (import does NOT run main(), it is guarded)
try:
    import bidmc_full53 as B
except Exception as e:
    raise SystemExit(f"could not import bidmc_full53.py (run in same folder): {e}")

SEED = B.SEED
FS_WORK = B.FS_WORK
SEG = B.SEG_SECONDS
N_SEEDS = B.N_PHASE2_SEEDS
CONTAM = B.CONTAM


def load_clouds():
    """Rebuild the same 53 three-channel clouds the original analysis used."""
    ids = B.all_record_ids()
    clouds = []
    print(f"Loading {len(ids)} BIDMC records (same pipeline as bidmc_full53)...")
    for i, rid in enumerate(ids, 1):
        d, status = B.load_record(rid)
        if d is None:
            print(f"[{i:2d}/{len(ids)}] {rid}: {status}"); continue
        ppg = B.clean_and_resample(d["ppg"], d["fs_wave"], FS_WORK, SEG)
        resp = B.clean_and_resample(d["resp"], d["fs_wave"], FS_WORK, SEG)
        spo2 = B.clean_and_resample(d["spo2"], d["fs_num"], FS_WORK, SEG)
        if ppg is None:
            print(f"[{i:2d}/{len(ids)}] {rid}: no usable PPG"); continue
        ppg = B.bandpass(ppg, FS_WORK, 0.5, 4.0)
        hr = B.estimate_hr(ppg, FS_WORK)
        cloud = B.stacked_embedding(ppg, resp, spo2, FS_WORK, hr)
        if cloud is None:
            print(f"[{i:2d}/{len(ids)}] {rid}: embedding too short"); continue
        clouds.append((rid, cloud))
        print(f"[{i:2d}/{len(ids)}] {rid} loaded")
    return clouds


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation from signed differences.
    +1 => GTPS always better, -1 => always worse, 0 => no effect."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    Rpos = ranks[d > 0].sum()
    Rneg = ranks[d < 0].sum()
    tot = Rpos + Rneg
    return float((Rpos - Rneg) / tot) if tot > 0 else 0.0


def main():
    t0 = time.time()
    clouds = load_clouds()
    n = len(clouds)
    if n < 5:
        print("Too few clouds; check data."); return
    print(f"\nPhase 2 on {n} recordings x {N_SEEDS} realisations "
          f"(maxdim=2). Please wait...\n")

    # per-recording averaged pairs; and raw matrix
    per_rec = {k: {"std": [], "gtps": []} for k in (0, 1, 2)}
    raw_rows = []
    for j, (rid, cloud) in enumerate(clouds, 1):
        clean = B.persistence(cloud, seed=SEED)
        sv = {k: [] for k in (0, 1, 2)}
        gv = {k: [] for k in (0, 1, 2)}
        for sdi in range(N_SEEDS):
            rng = np.random.default_rng(SEED + sdi)
            noisy = B.contaminate(cloud, CONTAM, rng)
            keep = B.gtps_keep(noisy)
            dstd = B.persistence(noisy, seed=SEED + sdi)
            dgt = B.persistence(noisy[keep] if keep.sum() >= 10 else noisy,
                                seed=SEED + sdi)
            for k in (0, 1, 2):
                a = B.bdist(clean[k], dstd[k])
                b = B.bdist(clean[k], dgt[k])
                sv[k].append(a); gv[k].append(b)
                raw_rows.append({"record": rid, "seed": sdi, "degree": k,
                                 "std_bottleneck": a, "gtps_bottleneck": b})
        for k in (0, 1, 2):
            per_rec[k]["std"].append(np.nanmean(sv[k]))
            per_rec[k]["gtps"].append(np.nanmean(gv[k]))
        if j % 5 == 0:
            print(f"  phase2 {j}/{n} ({time.time()-t0:.0f}s)")

    # save raw matrix for future re-analysis (no recompute needed later)
    with open("bidmc_bottleneck_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["record", "seed", "degree",
                                          "std_bottleneck", "gtps_bottleneck"])
        w.writeheader(); w.writerows(raw_rows)

    # ---- corrected statistics ----
    print("\n" + "=" * 78)
    print("CORRECTED STATS  (n = {} recordings; 5 realisations averaged within "
          "recording)".format(n))
    print("Wilcoxon signed-rank (paired); effect = matched-pairs rank-biserial")
    print("=" * 78)
    hdr = (f"{'deg':<4}{'Std mean':>10}{'GTPS mean':>11}{'med diff':>10}"
           f"{'% reduc':>9}{'Wilcoxon p':>12}{'effect r':>10}")
    print(hdr)
    summary = [hdr]
    for k, nm in [(0, "H0"), (1, "H1"), (2, "H2")]:
        std_a = np.array(per_rec[k]["std"])
        gtps_a = np.array(per_rec[k]["gtps"])
        diff = std_a - gtps_a          # >0 => GTPS smaller bottleneck (better)
        try:
            _, p = wilcoxon(std_a, gtps_a)
        except Exception:
            p = 1.0
        med = float(np.median(diff))
        base = float(np.median(std_a))
        pct = 100.0 * med / base if base != 0 else 0.0
        r_rb = rank_biserial(diff)
        line = (f"{nm:<4}{std_a.mean():>10.3f}{gtps_a.mean():>11.3f}"
                f"{med:>10.4f}{pct:>8.1f}%{p:>12.2g}{r_rb:>10.2f}")
        print(line); summary.append(line)

    print("\nInterpretation guide:")
    print("  - Wilcoxon p is the honest significance with n=53 (no pseudoreplication).")
    print("  - 'effect r' near +1 = GTPS reliably better; near 0 = negligible.")
    print("  - '% reduc' is the median bottleneck reduction; small values ARE small.")
    with open("bidmc_corrected_stats.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nSaved bidmc_bottleneck_raw.csv and bidmc_corrected_stats.txt "
          f"(total {time.time()-t0:.0f}s).")
    print("Paste the CORRECTED STATS block back.")


if __name__ == "__main__":
    main()
