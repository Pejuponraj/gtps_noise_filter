"""
bidmc_h2_surrogate.py
=====================
Addresses Prof. Muthu's point #4: the H2 (cardiac-autonomic void) significance was
argued against an arbitrary binomial null. The proper test for TDA-on-time-series
is a SURROGATE-DATA test: generate surrogates that preserve the signal's power
spectrum (and amplitude distribution) but destroy nonlinear phase structure, then
check whether the OBSERVED H2 persistence exceeds what spectrally-matched noise
produces. If it does, the void is a genuine nonlinear feature, not a spectral
artifact.

We use IAAFT surrogates (Iterative Amplitude Adjusted Fourier Transform, Schreiber
& Schmitz 1996): each channel's surrogate has the same amplitude histogram and the
same power spectrum as the original, but randomized Fourier phases. The stacked
embedding + persistent homology is recomputed on the surrogates, and the observed
H2 max-persistence is compared to the surrogate distribution (one-sided rank test).

Per recording we report the observed H2, the surrogate mean, and an empirical
p-value = (1 + #surrogates >= observed) / (1 + N_surrogates). We then summarise
across recordings: how many have p < 0.05, and the median observed-vs-surrogate
ratio.

Reuses bidmc_full53.py (import). Run in the SAME folder:
    pip install scipy numpy ripser persim wfdb
    python bidmc_h2_surrogate.py
Paste the SURROGATE RESULT block back.
"""

from __future__ import annotations
import csv
import time
import numpy as np

try:
    from scipy.spatial.distance import cdist
except Exception:
    raise SystemExit("scipy needed -> pip install scipy")
try:
    from ripser import ripser
except Exception:
    raise SystemExit("ripser needed -> pip install ripser")

try:
    import bidmc_full53 as B
except Exception as e:
    raise SystemExit(f"import bidmc_full53.py failed (run in same folder): {e}")

SEED = B.SEED
FS_WORK = B.FS_WORK
SEG = B.SEG_SECONDS
MAX_PTS = getattr(B, "MAX_PTS", 200)
N_SURR = 19          # surrogates per recording (19 -> resolution 0.05 one-sided)
N_ITER = 100         # IAAFT iterations


def subsample(cloud, seed):
    if len(cloud) <= MAX_PTS:
        return cloud
    idx = np.random.default_rng(seed).choice(len(cloud), MAX_PTS, replace=False)
    return cloud[idx]


def h2max(dg):
    d = dg[2]
    d = d[np.isfinite(d[:, 1])] if len(d) else d
    return float((d[:, 1] - d[:, 0]).max()) if len(d) else 0.0


def iaaft(x, n_iter=N_ITER, rng=None):
    """IAAFT surrogate of 1D signal x: same amplitude distribution and (approx)
    same power spectrum, randomized phases."""
    if rng is None:
        rng = np.random.default_rng(0)
    if x is None:
        return None
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = int(x.size)
    if n < 8:
        return x
    amp = np.abs(np.fft.rfft(x))            # target amplitude spectrum
    sorted_x = np.sort(x)                    # target amplitude distribution
    # start from a random permutation
    s = rng.permutation(x)
    for _ in range(n_iter):
        # match power spectrum
        S = np.fft.rfft(s)
        phase = np.angle(S)
        s = np.fft.irfft(amp * np.exp(1j * phase), n=n)
        # match amplitude distribution (rank-remap onto sorted_x)
        ranks = np.argsort(np.argsort(s))
        s = sorted_x[ranks]
    return s


def build_cloud_from_channels(ppg, resp, spo2, hr):
    return B.stacked_embedding(ppg, resp, spo2, FS_WORK, hr)


def load_channels():
    """Return list of (rid, ppg, resp, spo2, hr) so we can surrogate each channel."""
    ids = B.all_record_ids()
    out = []
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
        # some BIDMC records lack a usable resp/SpO2 trace -> cannot surrogate; skip
        def _ok(v):
            return v is not None and np.atleast_1d(np.asarray(v)).size >= 8
        if not (_ok(resp) and _ok(spo2)):
            continue
        ppg = B.bandpass(ppg, FS_WORK, 0.5, 4.0)
        hr = B.estimate_hr(ppg, FS_WORK)
        if build_cloud_from_channels(ppg, resp, spo2, hr) is not None:
            out.append((rid, ppg, resp, spo2, hr))
    print(f"Loaded {len(out)} recordings.\n")
    return out


def main():
    print("=" * 60)
    print("  bidmc_h2_surrogate  VERSION 2  (missing-channel safe)")
    print("=" * 60)
    t0 = time.time()
    recs = load_channels()
    n = len(recs)
    if n < 5:
        print("Too few recordings."); return
    print(f"IAAFT surrogate test for H2 on {n} recordings "
          f"({N_SURR} surrogates each). Please wait...\n")

    rows = []
    n_sig = 0
    ratios = []
    for j, (rid, ppg, resp, spo2, hr) in enumerate(recs, 1):
        cloud = build_cloud_from_channels(ppg, resp, spo2, hr)
        obs = h2max(ripser(subsample(cloud, SEED), maxdim=2)["dgms"])
        surr_vals = []
        for si in range(N_SURR):
            rng = np.random.default_rng(SEED + 1000 * j + si)
            try:
                sp = iaaft(ppg, rng=rng)
                sr = iaaft(resp, rng=rng)
                ss = iaaft(spo2, rng=rng)
                sc = build_cloud_from_channels(sp, sr, ss, hr)
            except Exception:
                continue          # unusable channel in this record -> skip surrogate
            if sc is None:
                continue
            surr_vals.append(h2max(ripser(subsample(sc, SEED), maxdim=2)["dgms"]))
        surr_vals = np.array(surr_vals) if surr_vals else np.array([0.0])
        # one-sided empirical p-value
        p = (1 + int((surr_vals >= obs).sum())) / (1 + len(surr_vals))
        ratio = obs / (surr_vals.mean() + 1e-9)
        if p < 0.05:
            n_sig += 1
        ratios.append(ratio)
        rows.append({"record": rid, "obs_H2": round(obs, 4),
                     "surr_mean": round(float(surr_vals.mean()), 4),
                     "ratio": round(ratio, 3), "p": round(p, 4)})
        if j % 5 == 0:
            print(f"  {j}/{n} ({time.time()-t0:.0f}s)")

    with open("bidmc_h2_surrogate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["record", "obs_H2", "surr_mean", "ratio", "p"])
        w.writeheader(); w.writerows(rows)

    ratios = np.array(ratios)
    print("\n" + "=" * 72)
    print(f"SURROGATE RESULT for H2 (IAAFT, {N_SURR} surrogates/recording)")
    print("=" * 72)
    print(f"  Recordings tested            : {n}")
    print(f"  H2 significant (p < 0.05)    : {n_sig}/{n} "
          f"({100.0*n_sig/n:.0f}%)")
    print(f"  Median observed/surrogate ratio : {np.median(ratios):.2f} "
          f"(>1 means observed H2 exceeds spectrally-matched noise)")
    print(f"  Recordings with ratio > 1    : {int((ratios>1).sum())}/{n}")
    print("\nInterpretation:")
    print("  * A high fraction with p<0.05 and ratio>1 => the H2 void is a genuine")
    print("    nonlinear feature, not reproducible by spectrally-matched noise.")
    print("  * This replaces the arbitrary binomial null with a proper surrogate test.")
    print("  * Whatever the numbers are is what we report.")
    print(f"\nSaved bidmc_h2_surrogate.csv. Total {time.time()-t0:.0f}s. Paste the block.")


if __name__ == "__main__":
    main()
