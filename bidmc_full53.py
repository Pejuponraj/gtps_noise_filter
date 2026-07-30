"""
bidmc_full53.py  --  full 53-record real-data study (BIDMC / PhysioNet).

Extends the 8-record test to ALL 53 recordings, adds H2 detection-threshold
sensitivity, saves a per-record CSV, and runs the GTPS-vs-TDA comparison with a
progress readout. This is the foundation for the real-data (Q1) claims.

Outputs (written next to the script):
    bidmc_per_record.csv     one row per record: HR, H1/H2 lifetimes & ratios
    bidmc_summary.txt        the printed summary (detection + GTPS comparison)

Integrity: fixed seeds; every number is computed, none hand-edited; records where
a feature is absent or GTPS does not help are reported as-is.

Run:
    python bidmc_full53.py
Time: maxdim=2 over 53 records is slow -- expect ~30-60 min. Progress is printed
per record so you can see it is working.
"""

from __future__ import annotations
import os
import csv
import time
import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.spatial.distance import squareform, pdist
from scipy.stats import ttest_rel

import wfdb
from ripser import ripser
from persim import bottleneck

SEED = 42
FS_WORK = 30.0
SEG_SECONDS = 30.0
MAX_PTS = 200          # subsample cap before ripser (H2 is expensive)
N_PHASE2_SEEDS = 5     # contamination repetitions per record in Phase 2
CONTAM = 0.8
OUT_CSV = "bidmc_per_record.csv"
OUT_TXT = "bidmc_summary.txt"


# --------------------------------------------------------------------------- #
def all_record_ids():
    try:
        return wfdb.get_record_list("bidmc")
    except Exception:
        return [f"bidmc{i:02d}" for i in range(1, 54)]


DL_DIR = os.path.abspath("bidmc_data")  # absolute so load matches save location
_LOCAL_MAP = {}                          # record_name -> full path (no extension)


def _build_local_map():
    """Walk DL_DIR and map every <name>.hea to its full path, whatever the
    subfolder layout wfdb used. Bulletproof against nested directories."""
    _LOCAL_MAP.clear()
    for root, _, files in os.walk(DL_DIR):
        for fn in files:
            if fn.endswith(".hea"):
                _LOCAL_MAP[fn[:-4]] = os.path.join(root, fn[:-4])


def ensure_downloaded(ids):
    """Download the whole BIDMC database once into DL_DIR (skips if present),
    then index the files on disk."""
    already = os.path.isdir(DL_DIR) and any(
        f.endswith(".hea") for _, _, fs in os.walk(DL_DIR) for f in fs)
    if not already:
        print(f"Downloading BIDMC to {DL_DIR} (one time, a few minutes)...")
        try:
            wfdb.dl_database("bidmc", dl_dir=DL_DIR)
            print("Download complete.")
        except Exception as e:
            print(f"Bulk download failed ({str(e)[:60]}); will stream per record.")
    _build_local_map()
    hea = len(_LOCAL_MAP)
    print(f"Indexed {hea} .hea files on disk in {DL_DIR}.")
    if hea == 0:
        print("  (none found locally -> falling back to streaming from PhysioNet)")
    print()


def load_record(rid):
    # prefer local indexed file; fall back to streaming
    local = _LOCAL_MAP.get(rid)
    localn = _LOCAL_MAP.get(rid + "n")
    try:
        if local and localn:
            rec = wfdb.rdrecord(local)
            recn = wfdb.rdrecord(localn)
        else:
            rec = wfdb.rdrecord(rid, pn_dir="bidmc/1.0.0")
            recn = wfdb.rdrecord(rid + "n", pn_dir="bidmc/1.0.0")
    except Exception as e:
        return None, f"load failed: {str(e)[:60]}"

    def col(record, *names):
        clean = [s.strip().strip(",").strip().upper() for s in record.sig_name]
        for nm in names:
            k = nm.strip().upper()
            if k in clean:
                return record.p_signal[:, clean.index(k)]
        return None

    return {
        "ppg": col(rec, "PLETH"), "resp": col(rec, "RESP"),
        "fs_wave": rec.fs, "spo2": col(recn, "SpO2"), "fs_num": recn.fs,
    }, "ok"


def clean_and_resample(x, fs_in, fs_out, seg_s, start_frac=0.3):
    if x is None:
        return None
    x = np.asarray(x, float); n = len(x)
    a = int(start_frac * n); b = min(a + int(seg_s * fs_in), n)
    seg = x[a:b]
    if len(seg) < 10 or np.all(np.isnan(seg)):
        return None
    seg = np.nan_to_num(seg, nan=np.nanmean(seg))
    m = int(round(len(seg) * fs_out / fs_in))
    return resample(seg, m) if m >= 8 else None


def bandpass(x, fs, lo, hi):
    nyq = 0.5 * fs
    b, a = butter(3, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
    return filtfilt(b, a, x)


def estimate_hr(ppg, fs, lo=0.7, hi=3.0):
    x = ppg - ppg.mean()
    f = np.fft.rfftfreq(len(x), 1 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    band = (f >= lo) & (f <= hi)
    return float(f[band][np.argmax(p[band])] * 60) if band.any() else 72.0


def z(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-9)


def stacked_embedding(ppg, resp, spo2, fs, hr):
    tau = max(int(round(fs * 60.0 / hr / 4)), 1)
    chans, dims = [z(ppg)], [3]
    if resp is not None:
        chans.append(z(resp)); dims.append(2)
    if spo2 is not None:
        chans.append(z(spo2)); dims.append(2)
    m = min(len(c) for c in chans) - (max(dims) - 1) * tau
    if m < 12:
        return None
    cols = [np.stack([c[i * tau:i * tau + m] for i in range(d)], 1)
            for c, d in zip(chans, dims)]
    return np.concatenate(cols, 1)


def persistence(cloud, maxdim=2, seed=0):
    c = np.asarray(cloud, float)
    if len(c) > MAX_PTS:
        idx = np.random.default_rng(seed).choice(len(c), MAX_PTS, replace=False)
        c = c[idx]
    dgms = ripser(c, maxdim=maxdim)["dgms"]
    out = []
    for d in dgms:
        d = np.asarray(d, float)
        if d.size:
            fin = d[np.isfinite(d[:, 1])]
            cap = fin[:, 1].max() if fin.size else 1.0
            d = d.copy(); d[~np.isfinite(d[:, 1]), 1] = cap
        out.append(d)
    while len(out) < 3:
        out.append(np.empty((0, 2)))
    return out


def prominence(dgm):
    if dgm is None or len(dgm) == 0:
        return 0.0, 0.0
    life = dgm[:, 1] - dgm[:, 0]
    longest = float(life.max()); floor = float(np.median(life))
    return longest, (longest / floor if floor > 1e-9 else float("inf"))


def bdist(a, b):
    a = a if a is not None and len(a) else np.empty((0, 2))
    b = b if b is not None and len(b) else np.empty((0, 2))
    try:
        return float(bottleneck(a, b))
    except Exception:
        return float("nan")


def gtps_keep(cloud, c=1.0, k=6):
    D = squareform(pdist(cloud)); n = len(cloud)
    knn = np.sort(D, 1)[:, 1:min(k, n - 1) + 1].mean(1)
    return knn <= knn.mean() + c * knn.std()


def contaminate(cloud, frac, rng):
    lo, hi = cloud.min(0), cloud.max(0)
    noise = rng.uniform(lo, hi, (int(frac * len(cloud)), cloud.shape[1]))
    return np.vstack([cloud, noise])


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ids = all_record_ids()
    ensure_downloaded(ids)
    print(f"Found {len(ids)} BIDMC records. Analysing all (maxdim=2, be patient)...\n")
    rows = []            # per-record detection results
    clouds = []          # (rid, cloud) for phase 2
    lines = []           # captured summary text

    def emit(s=""):
        print(s); lines.append(s)

    emit("record   HR   H1_life H1_ratio  H2_life H2_ratio  verdict")
    try:
        for i, rid in enumerate(ids, 1):
            d, status = load_record(rid)
            if d is None:
                print(f"[{i:2d}/{len(ids)}] {rid}: {status}")
                continue
            ppg = clean_and_resample(d["ppg"], d["fs_wave"], FS_WORK, SEG_SECONDS)
            resp = clean_and_resample(d["resp"], d["fs_wave"], FS_WORK, SEG_SECONDS)
            spo2 = clean_and_resample(d["spo2"], d["fs_num"], FS_WORK, SEG_SECONDS)
            if ppg is None:
                print(f"[{i:2d}/{len(ids)}] {rid}: no usable PPG segment")
                continue
            ppg = bandpass(ppg, FS_WORK, 0.5, 4.0)
            hr = estimate_hr(ppg, FS_WORK)
            cloud = stacked_embedding(ppg, resp, spo2, FS_WORK, hr)
            if cloud is None:
                print(f"[{i:2d}/{len(ids)}] {rid}: embedding too short")
                continue
            dg = persistence(cloud, seed=SEED)
            h1l, h1r = prominence(dg[1]); h2l, h2r = prominence(dg[2])
            h1_ok = h1l > 0.3 and h1r > 4.0
            h2_ok = h2l > 0.2 and h2r > 3.0
            verdict = ("H1+H2" if h1_ok and h2_ok else
                       "H1 only" if h1_ok else
                       "H2 only" if h2_ok else "none")
            emit(f"{rid:<8}{hr:>4.0f}  {h1l:>8.3f}{h1r:>8.2f}  {h2l:>8.3f}{h2r:>8.2f}  {verdict}")
            rows.append({"record": rid, "HR": round(hr, 1),
                         "H1_life": round(h1l, 4), "H1_ratio": round(h1r, 3),
                         "H2_life": round(h2l, 4), "H2_ratio": round(h2r, 3),
                         "H1_detected": int(h1_ok), "H2_detected": int(h2_ok)})
            clouds.append((rid, cloud))
            print(f"[{i:2d}/{len(ids)}] {rid} done ({time.time()-t0:.0f}s elapsed)")
    except KeyboardInterrupt:
        emit(f"\n[interrupted after {len(rows)} records -- saving partial results]")

    # ---- detection summary + threshold sensitivity ----
    n = len(rows)
    if n == 0:
        emit("\nNo records analysed."); return
    h1 = sum(r["H1_detected"] for r in rows)
    emit(f"\nDETECTION (n={n} records):")
    emit(f"  H1 loop: {h1}/{n} ({100*h1/n:.0f}%)")
    emit("  H2 void at several thresholds (life>L and ratio>R):")
    for L, R in [(0.15, 2.5), (0.20, 3.0), (0.25, 4.0)]:
        h2 = sum(1 for r in rows if r["H2_life"] > L and r["H2_ratio"] > R)
        emit(f"    L>{L}, R>{R}: {h2}/{n} ({100*h2/n:.0f}%)")

    # ---- Phase 2: GTPS vs TDA on real clouds ----
    emit("\nPHASE 2  --  GTPS vs Standard TDA on real clouds (+contamination)")
    res = {k: {"std": [], "gtps": []} for k in (0, 1, 2)}
    for j, (rid, cloud) in enumerate(clouds, 1):
        clean = persistence(cloud, seed=SEED)
        for s in range(N_PHASE2_SEEDS):
            rng = np.random.default_rng(SEED + s)
            noisy = contaminate(cloud, CONTAM, rng)
            keep = gtps_keep(noisy)
            dstd = persistence(noisy, seed=SEED + s)
            dgt = persistence(noisy[keep] if keep.sum() >= 10 else noisy, seed=SEED + s)
            for k in (0, 1, 2):
                res[k]["std"].append(bdist(clean[k], dstd[k]))
                res[k]["gtps"].append(bdist(clean[k], dgt[k]))
        if j % 5 == 0:
            print(f"  phase2 {j}/{len(clouds)} records ({time.time()-t0:.0f}s)")
    emit(f"{'degree':<7}{'Std-TDA':>9}{'GTPS':>9}{'better':>9}{'p':>11}")
    for k, nm in [(0, "H0"), (1, "H1"), (2, "H2")]:
        a, b = np.array(res[k]["std"]), np.array(res[k]["gtps"])
        ma, mb = np.nanmean(a), np.nanmean(b)
        try:
            p = ttest_rel(a, b, nan_policy="omit").pvalue
        except Exception:
            p = 1.0
        if np.isnan(p): p = 1.0
        win = "GTPS" if mb < ma - 1e-9 else ("tie" if abs(mb - ma) < 1e-9 else "Std")
        emit(f"{nm:<7}{ma:>9.3f}{mb:>9.3f}{win:>9}{p:>11.2g}{' *' if p < 0.05 else ''}")

    emit(f"\nTotal time: {time.time()-t0:.0f}s")

    # ---- write files ----
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines))
    emit(f"\nSaved: {OUT_CSV} (per-record) and {OUT_TXT} (summary).")
    emit("Paste the summary back so we can write the paper's real-data table.")


if __name__ == "__main__":
    main()