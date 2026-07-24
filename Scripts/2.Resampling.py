import os
import glob
import numpy as np
import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "rawdata", "2nd Trial", "Multimodal_data")

# Gap threshold: intervals longer than this multiple of the median interval
# are flagged as gaps (gap_mask = 0). At ~20 Hz, 3× ≈ 144 ms.
GAP_MULTIPLIER = 3

TARGET_FS = 20.0   # device nominal sampling rate in Hz

# Near-duplicate threshold in milliseconds.
# Rows within this time of each other are treated as a single frame burst
# from the sensor software and the second row is removed.
DEDUP_THRESHOLD_MS = 5.0

# Columns that are discrete gain settings — use nearest-neighbour, not linear
TIA_COLS = [
    "TIA gain (515nm)", "TIA gain (635nm)", "TIA gain (465nm)",
    "TIA gain (740nm)", "TIA gain (770nm)", "TIA gain (810nm)", "TIA gain (850nm)",
]

# ── core functions ─────────────────────────────────────────────────────────────

def remove_duplicates(time_sec, dedup_ms=5.0):
    """
    Return a boolean keep-mask removing the second row of any near-duplicate
    timestamp pair (< dedup_ms ms apart).
    Caused by sensor software writing multiple rows per binary frame.
    """
    keep = np.ones(len(time_sec), dtype=bool)
    dts = np.diff(time_sec)
    dup_idx = np.where(dts < dedup_ms / 1000.0)[0]
    keep[dup_idx + 1] = False
    return keep


def estimate_fs(time_sec):
    """
    Estimate sampling frequency from median interval of clean timestamps.
    Self-adapting: works correctly for any session without manual tuning.
    """
    dts = np.diff(time_sec)
    dts_valid = dts[(dts > 0) & np.isfinite(dts)]
    return 1.0 / np.median(dts_valid)


def build_gap_mask(time_orig, t_uniform, gap_threshold_s):
    """
    Build binary mask over the uniform grid:
      1 = interpolated from real nearby samples  → trustworthy
      0 = falls inside a genuine gap             → flag at GLM stage
    """
    dts = np.diff(time_orig)
    gap_starts = time_orig[:-1][dts > gap_threshold_s]
    gap_ends   = time_orig[1:] [dts > gap_threshold_s]

    mask = np.ones(len(t_uniform), dtype=float)
    for gs, ge in zip(gap_starts, gap_ends):
        mask[(t_uniform > gs) & (t_uniform < ge)] = 0.0
    return mask


def resample_file(fpath, gap_multiplier=GAP_MULTIPLIER,
                  dedup_ms=DEDUP_THRESHOLD_MS,
                  trim_to_events=True):
    """
    Full resampling pipeline for one CSV file.
    Output columns are identical to input columns + gap_mask at the end.
    Returns a dict of QC metrics.
    """
    fname = os.path.basename(fpath)
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")

    # 1. Load
    df = pd.read_csv(fpath)
    original_columns = list(df.columns)   # preserve exact column order

    # 2. Parse timestamps
    t_raw = pd.to_datetime(df["timestamp"], errors="coerce")
    t0_wall = t_raw.iloc[0]               # original wall-clock start (for reconstruction)
    time_sec_raw = (t_raw - t0_wall).dt.total_seconds().to_numpy()

    # 2b. Drop rows with unparseable (NaN) timestamps
    finite_mask = np.isfinite(time_sec_raw)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        print(f"  Removed {n_bad} rows with invalid/unparseable timestamps")
        time_sec_raw = time_sec_raw[finite_mask]
        df = df.iloc[np.where(finite_mask)[0]].reset_index(drop=True)

    # 3. Sort (safety)
    sort_idx = np.argsort(time_sec_raw)
    time_sec_raw = time_sec_raw[sort_idx]
    df = df.iloc[sort_idx].reset_index(drop=True)

    # 4. Remove near-duplicate timestamps
    keep = remove_duplicates(time_sec_raw, dedup_ms)
    time_sec = time_sec_raw[keep]
    df_clean = df.iloc[keep].reset_index(drop=True)
    n_dupes_removed = int((~keep).sum())
    print(f"  Rows removed (near-duplicates): {n_dupes_removed} / {len(df)}")

    # 5a. Trim to labelled window (drop pre-recording and post-recording tail)
    #     Looks for a matching *_events.csv beside the data file.
    #     If found: clips to [baseline_start, experiment_end / last rest_end].
    #     If not found: skips trim silently.
    if trim_to_events:
        events_path = fpath.replace(".csv", "_events.csv")
        # also try {stem}_events.csv pattern (e.g. Yasmine_Task → yasmine_events)
        if not os.path.exists(events_path):
            stem = os.path.splitext(os.path.basename(fpath))[0].lower().split("_task")[0]
            events_path = os.path.join(os.path.dirname(fpath), f"{stem}_events.csv")
        if os.path.exists(events_path):
            ev_df = pd.read_csv(events_path)
            ev_df["_ts"] = pd.to_datetime(ev_df["timestamp_wall_clock"], errors="coerce")
            # trim start: first labelled event (baseline_start if present, else first cue_start)
            start_rows = ev_df[ev_df["event_type"].isin(["baseline_start", "cue_start"])]
            # trim end: experiment_end if present, else last rest_end
            end_rows = ev_df[ev_df["event_type"].isin(["experiment_end", "rest_end"])]
            if not start_rows.empty and not end_rows.empty:
                t_trim_start = (start_rows["_ts"].min() - t0_wall).total_seconds()
                t_trim_end   = (end_rows["_ts"].max()   - t0_wall).total_seconds()
                before = len(time_sec)
                trim_mask = (time_sec >= t_trim_start) & (time_sec <= t_trim_end)
                time_sec  = time_sec[trim_mask]
                df_clean  = df_clean.iloc[trim_mask].reset_index(drop=True)
                trimmed_start = int((~trim_mask).cumsum().searchsorted(1))
                trimmed_end   = int(before - len(time_sec) - trimmed_start)
                print(f"  Trimmed {trimmed_start} pre-recording + {trimmed_end} post-recording samples")
                print(f"  Trimmed window: {t_trim_start:.2f}s – {t_trim_end:.2f}s")
        else:
            print(f"  [trim] No events file found alongside data — skipping trim")

    # 5. Estimate fs and gap threshold
    fs_estimated = estimate_fs(time_sec)
    fs = TARGET_FS
    print(f"  Raw estimated fs: {fs_estimated:.2f} Hz  |  Using fixed target: {fs} Hz")
    gap_threshold_s = gap_multiplier / fs
    print(f"  Estimated fs: {fs:.2f} Hz  |  gap threshold: {gap_threshold_s*1000:.0f} ms")

    # 6. Detect gaps for QC
    dts_all = np.diff(time_sec)
    gap_durations = dts_all[dts_all > gap_threshold_s]
    n_gaps = len(gap_durations)
    total_gap_s = float(gap_durations.sum()) if n_gaps > 0 else 0.0
    recording_duration_s = float(time_sec[-1] - time_sec[0])
    print(f"  Gaps detected: {n_gaps}  |  cumulative gap time: {total_gap_s:.2f} s "
          f"({100*total_gap_s/recording_duration_s:.1f}% of recording)")

    # 7. Build uniform time grid
    t_uniform = np.arange(time_sec[0], time_sec[-1], 1.0 / fs)
    print(f"  Uniform grid: {len(t_uniform)} samples at {fs:.2f} Hz")

    # 8. Build output dataframe — same column order as input
    df_out = pd.DataFrame(index=range(len(t_uniform)))

    for col in original_columns:

        # ── timestamp: reconstruct ISO strings from uniform grid ──────────────
        if col == "timestamp":
            reconstructed_times = pd.to_datetime(
                t0_wall.value + (t_uniform * 1e9).astype(np.int64)
            )
            df_out["timestamp"] = reconstructed_times.strftime("%Y-%m-%dT%H:%M:%S.%f")
            continue

        # ── event: always 0 in raw data, alignment script fills it later ──────
        if col == "event":
            df_out["event"] = 0
            continue

        # ── event_name: always empty in raw, alignment script fills it later ──
        if col == "event_name":
            df_out["event_name"] = ""
            continue

        # ── TIA gain columns: discrete settings, nearest-neighbour ────────────
        if col in TIA_COLS:
            x = df_clean[col].to_numpy(dtype=float)
            finite_mask = np.isfinite(x) & np.isfinite(time_sec)
            if finite_mask.sum() > 1:
                idx_nn = np.searchsorted(time_sec[finite_mask], t_uniform).clip(
                    0, finite_mask.sum() - 1
                )
                df_out[col] = x[finite_mask][idx_nn].astype(int)
            else:
                df_out[col] = 0
            continue

        # ── all other numeric columns: linear interpolation ───────────────────
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            x = df_clean[col].to_numpy(dtype=float)
            finite_mask = np.isfinite(x) & np.isfinite(time_sec)
            if finite_mask.sum() > 10:
                df_out[col] = np.interp(
                    t_uniform,
                    time_sec[finite_mask],
                    x[finite_mask]
                )
            else:
                df_out[col] = np.nan
            continue

        # ── any other non-numeric column: fill empty (safety) ─────────────────
        df_out[col] = ""

    # 9. Gap mask — appended at the end
    df_out["gap_mask"] = build_gap_mask(time_sec, t_uniform, gap_threshold_s)
    pct_valid = float(df_out["gap_mask"].mean()) * 100
    print(f"  Valid (non-gap) samples: {pct_valid:.1f}%")

    # 10. Save
    out_path = fpath.replace(".csv", "_resampled.csv")
    df_out.to_csv(out_path, index=False)
    print(f"  {fname}: {len(t_uniform)} samples, {n_gaps} gaps ({100*total_gap_s/recording_duration_s:.1f}%), saved ✅")


    return {
        "file": fname,
        "n_raw_rows": len(df),
        "n_dupes_removed": n_dupes_removed,
        "fs_hz": round(fs, 2),
        "recording_duration_s": round(recording_duration_s, 1),
        "n_gaps": n_gaps,
        "total_gap_s": round(total_gap_s, 2),
        "pct_gap": round(100 - pct_valid, 1),
        "n_resampled_rows": len(t_uniform),
    }


def print_qc_report(qc_records):
    """Print compact QC summary table."""
    print("\n" + "="*75)
    print("  fNIRS RESAMPLING SUMMARY")
    print("="*75)
    print(f"  {'Subject':<15} {'RawRows':>8} {'Dupes%':>7} {'Gaps':>6} {'GapTime%':>9} {'ResRows':>8}  Flags")
    print("  " + "-"*70)
    for r in qc_records:
        subject = r['file'].replace('_signals.csv','')
        dupe_pct = round(r['n_dupes_removed'] / r['n_raw_rows'] * 100, 1)
        flags = []
        if r['pct_gap'] > 5.0:
            flags.append(" high gap")
        if r['n_gaps'] > 50:
            flags.append(" jitter")
        flag_str = "  ".join(flags) if flags else "✅"
        print(f"  {subject:<15} {r['n_raw_rows']:>8} {dupe_pct:>6}% {r['n_gaps']:>6} "
              f"{r['pct_gap']:>8}%  {r['n_resampled_rows']:>8}  {flag_str}")
    print("="*75)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    files = []
    for root, dirs, fnames in os.walk(DATA_DIR):
        for f in sorted(fnames):
            if f.endswith("_signals.csv") and "resampled" not in f.lower():
                files.append(os.path.join(root, f))

    if not files:
        print(f"No *_signals.csv files found under: {DATA_DIR}")
        raise SystemExit

    print(f"Found {len(files)} subject file(s) to process:")
    for f in files:
        print(f"  {f}")

    qc_records = []
    for fpath in files:
        try:
            qc = resample_file(fpath)
            qc_records.append(qc)
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(fpath)}: {e}")

    print_qc_report(qc_records)
    print(f"\nDone. {len(qc_records)} file(s) resampled.")