"""
5a_Channel_Selection.py
========================
Stage 1 of the (split) preprocessing pipeline — channel selection.

  - Load aligned fNIRS data (already trimmed) + quality metrics
  - Drop failed PDs, keep 740+850nm for long-sep, all wavelengths for PD0
  - Coupling-shift check on PD0 (short channel) — drop it if unstable

Input  (per subject, from step 4):
    {subject}_fnirs_aligned.csv
    {subject}_quality_metrics.csv

Output (per subject):
    {subject}_channel_selected.csv   → consumed by 5b.Motion_Correction.py

Run:
    python "5a.Channel_Selection.py"
"""

import os
import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_SCRIPT_DIR, "..", "rawdata", "2nd Trial", "Multimodal_data")

ANALYSIS_WLS = [740, 850]          # wavelengths for Beer-Lambert
PHOTODIODES  = (0, 1, 2, 3)

# ── Coupling shift correction config (mirrors 4.Quality+PPG.py) ──
COUPLING_THRESH    = 0.05          # V — minimum step size to count as a coupling shift
COUPLING_DETECT_WL = 810           # wavelength for detection (810nm ≈ isosbestic point)


# ── Coupling shift correction helpers ────────────────────────────────────────
def detect_coupling_shifts(df, pd_idx, Fs, thresh_override=None):
    """
    Detect persistent coupling shifts in the COUPLING_DETECT_WL (810nm) channel
    of a given PD.  Mirrors the logic in 4.Quality+PPG.py: detect_all_shifts().

    Uses gap_mask to restrict detection to the valid experiment window and
    excludes the first/last 30s to avoid edge artifacts.
    Returns a sorted list of shift sample indices (empty if none found).

    thresh_override: use a different threshold instead of COUPLING_THRESH —
        useful for residual re-detection after correction, where a lenient
        threshold avoids false-failing small residuals.
    """
    col = f"fNIRS ({COUPLING_DETECT_WL}nm) PD{pd_idx}"
    if col not in df.columns:
        return []

    valid = (df["gap_mask"] == 1).to_numpy() if "gap_mask" in df.columns \
            else np.ones(len(df), dtype=bool)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return []

    buffer       = int(30 * Fs)
    start_margin = valid_idx[0]  + buffer
    end_margin   = valid_idx[-1] - buffer

    thresh = thresh_override if thresh_override is not None else COUPLING_THRESH

    sig   = df[col].to_numpy(float)
    steps = np.abs(np.diff(sig))

    raw_shifts = [
        i for i in range(len(steps) - 1)
        if steps[i] > thresh
        and valid[i] and valid[i + 1]
        and start_margin <= i <= end_margin
    ]

    if not raw_shifts:
        return []

    # Cluster: keep first sample of each cluster (within 50 samples)
    clustered = [raw_shifts[0]]
    for s in raw_shifts[1:]:
        if s - clustered[-1] > 50:
            clustered.append(s)
    return clustered


def get_passing_pds(quality_path):
    """
    Read quality CSV, return list of passing PDs split into short/long-sep.
    Subject is included only if at least one long-sep PD (PD1/2/3) passes.
    """
    if not os.path.exists(quality_path):
        return None, "quality metrics file not found"

    metrics = pd.read_csv(quality_path)
    passing = metrics[
        (metrics["wl"] == 740) &
        (metrics["pass_quality"] == True)
    ]["pd"].tolist()

    long_sep  = [p for p in passing if p > 0]
    short_sep = [p for p in passing if p == 0]
    include   = len(long_sep) > 0

    return {
        "long_sep":     long_sep,
        "short_sep":    short_sep,
        "all_passing":  passing,
        "include":      include,
    }, None


# ── Main processing per subject ────────────────────────────────────────────────
def process_subject(subject_folder, subject):
    """Channel selection for one subject. Returns summary dict + dataframe."""

    # Find aligned + quality files (try exact case, then lowercase)
    aligned_path = os.path.join(subject_folder, f"{subject}_fnirs_aligned.csv")
    quality_path = os.path.join(subject_folder, f"{subject}_quality_metrics.csv")

    if not os.path.exists(aligned_path):
        aligned_path = os.path.join(subject_folder,
                                    f"{subject.lower()}_fnirs_aligned.csv")
    if not os.path.exists(quality_path):
        quality_path = os.path.join(subject_folder,
                                    f"{subject.lower()}_quality_metrics.csv")

    if not os.path.exists(aligned_path):
        return {"subject": subject, "status": "❌ aligned file not found",
                "passing_pds": [], "long_sep": []}, None
    if not os.path.exists(quality_path):
        return {"subject": subject, "status": "❌ quality metrics not found",
                "passing_pds": [], "long_sep": []}, None

    # Load (already trimmed to experiment window by timestamp alignment)
    df = pd.read_csv(aligned_path, low_memory=False)
    Fs = 20.0

    print(f"\n{'='*60}")
    print(f"  {subject}  ({len(df)} samples, {len(df)/Fs:.1f}s)")
    print(f"{'='*60}")

    # ── Step 1: Get passing PDs ──
    pd_info, err = get_passing_pds(quality_path)
    if err:
        return {"subject": subject, "status": f"❌ {err}",
                "passing_pds": [], "long_sep": []}, None

    if not pd_info["include"]:
        print(f"  ❌ EXCLUDED — no long-sep channel passed")
        return {"subject": subject,
                "status": "❌ excluded — no long-sep passed",
                "passing_pds": [], "long_sep": []}, None

    passing_pds = pd_info["all_passing"]
    long_sep    = pd_info["long_sep"]
    short_sep   = pd_info["short_sep"]

    print(f"  ✅ INCLUDED — passing PDs: {passing_pds}")

    # ── Step 2: Keep only passing channels ──
    # Long-sep: 740nm + 850nm (Beer-Lambert pair)
    # PD0 (short-sep): all wavelengths for SSR flexibility
    metadata_cols = [c for c in df.columns if not c.startswith("fNIRS")]
    keep_cols = metadata_cols.copy()

    for pd_idx in long_sep:
        for wl in ANALYSIS_WLS:
            col = f"fNIRS ({wl}nm) PD{pd_idx}"
            if col in df.columns:
                keep_cols.append(col)

    if short_sep:   # PD0 passed → keep all its wavelengths
        for wl in [740, 770, 810, 850]:
            col = f"fNIRS ({wl}nm) PD0"
            if col in df.columns:
                keep_cols.append(col)

    dropped_pds = [p for p in PHOTODIODES if p not in passing_pds]
    df = df[[c for c in dict.fromkeys(keep_cols) if c in df.columns]].copy()

    fnirs_cols = [c for c in df.columns if c.startswith("fNIRS")]
    print(f"  Kept PDs: {passing_pds}  |  Dropped PDs: {dropped_pds}")
    print(f"  fNIRS channels: {len(fnirs_cols)}")

    # ── Step 2.5: Coupling shift check on PD0 (short channel) ──────────────
    pd0_cols = [c for c in fnirs_cols if "PD0" in c]
    if pd0_cols:
        print(f"\n  Coupling Shift Check (PD0):")
        shift_indices = detect_coupling_shifts(df, pd_idx=0, Fs=20.0)
        if shift_indices:
            times_str = ", ".join(f"{s/20.0:.1f}s" for s in shift_indices)
            print(f"    ❌ {len(shift_indices)} shift(s) detected at: [{times_str}] — dropping PD0")
            df          = df.drop(columns=pd0_cols, errors="ignore")
            fnirs_cols  = [c for c in fnirs_cols if "PD0" not in c]
            passing_pds = [p for p in passing_pds if p != 0]
            short_sep   = []
        else:
            print(f"    ✅ No coupling shifts — PD0 retained for SSR")

    # ── Save channel-selected output to subject folder ─────────────────────
    out_path = os.path.join(subject_folder, f"{subject}_channel_selected.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    return {
        "subject":      subject,
        "status":       "✅ included",
        "passing_pds":  passing_pds,
        "long_sep":     long_sep,
        "n_samples":    len(df),
        "n_fnirs_cols": len(fnirs_cols),
    }, df


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(results):
    print(f"\n{'='*70}")
    print(f"  CHANNEL SELECTION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Subject':<12} {'Include':>8} {'Passing':>12} {'Channels':>9} {'Samples':>8}")
    print(f"  {'-'*55}")
    for r in results:
        included = "✅" if "included" in r["status"] else "❌"
        if "excluded" in r["status"] or "not found" in r["status"]:
            print(f"  {r['subject']:<12} {included:>8}  {r['status']}")
        else:
            pds_str = str(r.get("passing_pds", []))
            print(f"  {r['subject']:<12} {included:>8} {pds_str:>12} "
                  f"{r.get('n_fnirs_cols', 0):>9} {r.get('n_samples', 0):>8}")
    print(f"{'='*70}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    subject_folders = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not subject_folders:
        print(f"No subject folders found in: {DATA_DIR}")
        return

    print(f"\nFound {len(subject_folders)} subject folder(s)")
    print("Step 5a: Channel Selection\n")

    results = []

    for folder in subject_folders:
        subject = os.path.basename(folder)
        try:
            result, df = process_subject(folder, subject)
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {subject}: {e}")
            results.append({"subject": subject, "status": f"💥 ERROR: {e}",
                            "passing_pds": [], "long_sep": []})

    print_summary(results)


if __name__ == "__main__":
    main()
