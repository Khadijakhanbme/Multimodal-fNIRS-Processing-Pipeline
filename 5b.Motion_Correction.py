"""
5b_Motion_Correction.py
========================
Stage 2 of the (split) preprocessing pipeline — IMU-based motion detection
and targeted PCA correction on fNIRS intensity.

Input  (per subject, from 5a.Channel_Selection.py):
    {subject}_channel_selected.csv

Output (per subject):
    {subject}_motion_corrected.csv   → consumed by 5c.OD_Conversion_Bandpass.py

Run:
    python "5b.Motion_Correction.py"
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ── Config ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_SCRIPT_DIR, "..", "rawdata", "2nd Trial", "Multimodal_data")

# ── Motion detection config ──
HP_GRAV          = 0.3             # highpass for gravity removal
MOT_THRESH       = 4.0             # z-score threshold for motion
VAR_PC1_THRESH   = 0.30            # variance threshold for PCA correction
MIN_WIN_LEN      = 5               # minimum motion window length (samples)


# ── Signal processing helpers ────
def highpass(sig, lo, fs, order=3):
    """Butterworth highpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, lo / nyq, btype="high")
    return filtfilt(b, a, sig)


def expand_mask(mask, n_expand):
    """Expand True regions in boolean mask by n_expand samples on each side."""
    N      = len(mask)
    result = mask.copy()
    for i in np.where(mask)[0]:
        result[max(0, i - n_expand):min(N, i + n_expand + 1)] = True
    return result


def get_windows(mask):
    """Extract contiguous True regions from boolean mask as (start, end) tuples."""
    windows, N, i = [], len(mask), 0
    while i < N:
        if mask[i]:
            j = i + 1
            while j < N and mask[j]:
                j += 1
            windows.append((i, j))
            i = j
        else:
            i += 1
    return windows


def detect_and_correct_motion(df, fnirs_cols, Fs):
    """
    IMU-based motion detection + targeted PCA correction on fNIRS intensity.

    Returns: (intensity_corrected_dict, motion_stats_dict, debug_info)
    - intensity_corrected_dict: {col_name: corrected_signal}
    - motion_stats_dict: {motion_detected, n_windows, pca_applied, pca_skipped}
    - debug_info: {var_pc1_values, window_indices}
    """
    N = len(df)

    # ── IMU motion detection ──
    ax = df.get("IMU (ax)", pd.Series(np.zeros(N))).to_numpy(float)
    ay = df.get("IMU (ay)", pd.Series(np.zeros(N))).to_numpy(float)
    az = df.get("IMU (az)", pd.Series(np.zeros(N))).to_numpy(float)

    if ax.shape[0] == 0 or np.all(np.isnan(ax)):
        # No IMU data — return raw intensity
        intensity_dict = {col: df[col].to_numpy(float) for col in fnirs_cols}
        return intensity_dict, {"motion_detected": False, "n_windows": 0,
                               "pca_applied": 0, "pca_skipped": 0}, {}

    # Magnitude and jerk-based detection
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    mag_hp = highpass(mag, HP_GRAV, Fs)
    jerk = np.diff(mag_hp, prepend=mag_hp[0]) * Fs

    med_j = np.median(jerk)
    mad_j = np.median(np.abs(jerk - med_j)) * 1.4826
    z = (jerk - med_j) / mad_j if mad_j > 0 else np.zeros_like(jerk)

    motion_raw = np.abs(z) > MOT_THRESH
    n_expand = int(round(2.0 * Fs))
    motion_mask = expand_mask(motion_raw, n_expand)
    windows = get_windows(motion_mask)

    print(f"    Motion detection: {motion_raw.sum()} samples flagged, "
          f"{motion_mask.sum()} after expansion ({100.0*motion_mask.sum()/N:.1f}%)")

    # ── PCA correction on motion windows ──
    if len(windows) == 0:
        # No motion detected
        intensity_dict = {col: df[col].to_numpy(float) for col in fnirs_cols}
        return intensity_dict, {"motion_detected": False, "n_windows": 0,
                               "pca_applied": 0, "pca_skipped": 0}, {}

    # Build intensity matrix
    I_matrix = np.column_stack([df[col].to_numpy(float) for col in fnirs_cols])
    n_pca_applied = 0
    n_pca_skipped = 0
    var_pc1_values = []
    corrected_windows = []

    for i0, i1 in windows:
        win_len = i1 - i0
        if win_len < MIN_WIN_LEN:
            n_pca_skipped += 1
            var_pc1_values.append((i0, i1, None, "too_short"))
            continue

        X = I_matrix[i0:i1, :]
        means = X.mean(axis=0)
        X_dm = X - means

        U, S, Vt = np.linalg.svd(X_dm, full_matrices=False)
        var_pc1 = S[0]**2 / np.sum(S**2)

        if var_pc1 > VAR_PC1_THRESH:
            # Remove PC1 and reconstruct
            I_matrix[i0:i1, :] = U[:, 1:] @ np.diag(S[1:]) @ Vt[1:, :] + means
            n_pca_applied += 1
            var_pc1_values.append((i0, i1, var_pc1, "applied"))
            corrected_windows.append((i0, i1, var_pc1))
        else:
            n_pca_skipped += 1
            var_pc1_values.append((i0, i1, var_pc1, "skipped"))

    print(f"    PCA correction: {n_pca_applied} windows corrected, "
          f"{n_pca_skipped} skipped (var_pc1 ≤ {VAR_PC1_THRESH})")

    # Convert back to dict
    intensity_corrected = {col: I_matrix[:, i] for i, col in enumerate(fnirs_cols)}

    debug_info = {
        "var_pc1_values": var_pc1_values,
        "corrected_windows": corrected_windows,
        "motion_mask_sum": int(motion_mask.sum()),
    }

    return intensity_corrected, {"motion_detected": True, "n_windows": len(windows),
                                "pca_applied": n_pca_applied, "pca_skipped": n_pca_skipped}, debug_info


def plot_motion_correction(subject, df, df_corrected, fnirs_cols, Fs, debug_info):
    """
    Plot motion detection and correction results.
    Layout: 3×3 GridSpec.
      Row 0: zoomed channel 1 | zoomed channel 2 | PC1 variance bar chart
      Row 1: removed component | per-subject summary table | full corrected signal
      Row 2: cardiac visibility PSD (spans all 3 columns)
    """
    try:
        N = len(df)
        t = np.arange(N) / Fs

        plot_channels = [col for col in fnirs_cols if "740nm" in col][:2]
        if not plot_channels:
            return

        corrected_windows = debug_info.get("corrected_windows", [])
        var_pc1_values    = debug_info.get("var_pc1_values", [])

        if not corrected_windows:
            print(f"    ⚠️  No windows were corrected (all skipped). Check VAR_PC1_THRESH or MIN_WIN_LEN.")
            return

        largest_window = max(corrected_windows, key=lambda x: x[1] - x[0])
        i0, i1, var_pc1 = largest_window

        pad        = int(2 * Fs)
        zoom_start = max(0, i0 - pad)
        zoom_end   = min(N, i1 + pad)
        t_zoom     = t[zoom_start:zoom_end]

        # Full motion mask rebuilt from all detected windows (for cardiac check + % flagged)
        motion_mask = np.zeros(N, dtype=bool)
        for entry in var_pc1_values:
            motion_mask[entry[0]:entry[1]] = True

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(f"{subject} — Motion Correction Diagnostics",
                     fontsize=12, fontweight="bold")
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

        # ── Panel 1: Zoomed channel 1 ──────────────────────────────────────────
        ax1  = fig.add_subplot(gs[0, 0])
        col1 = plot_channels[0]
        ax1.plot(t_zoom, df[col1].to_numpy(float)[zoom_start:zoom_end],
                 "b--", label="Raw", linewidth=2, alpha=0.7)
        ax1.plot(t_zoom, df_corrected[col1].to_numpy(float)[zoom_start:zoom_end],
                 "g-", label="Corrected", linewidth=2)
        ax1.axvspan(t[i0], t[i1], alpha=0.2, color="red", label="(Zoomed to the largest Motion window)")
        ax1.set_ylabel("Intensity (V)", fontsize=11)
        ax1.set_title(f"{col1.replace('fNIRS ', '')} - Zoomed View", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Zoomed channel 2 ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        if len(plot_channels) > 1:
            col2 = plot_channels[1]
            ax2.plot(t_zoom, df[col2].to_numpy(float)[zoom_start:zoom_end],
                     "b--", label="Raw", linewidth=2, alpha=0.7)
            ax2.plot(t_zoom, df_corrected[col2].to_numpy(float)[zoom_start:zoom_end],
                     "g-", label="Corrected", linewidth=2)
            ax2.axvspan(t[i0], t[i1], alpha=0.2, color="red", label="Motion window")
            ax2.set_ylabel("Intensity (V)", fontsize=11)
            ax2.set_title(f"{col2.replace('fNIRS ', '')} - Zoomed View", fontsize=10, fontweight="bold")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Only one 740nm channel", ha="center", va="center", fontsize=12)
            ax2.axis("off")

        # ── Panel 3: PC1 variance bar chart ───────────────────────────────────
        ax3      = fig.add_subplot(gs[0, 2])
        pc1_vals = [x[2] for x in corrected_windows]
        n_wins   = len(pc1_vals)
        x_pos    = np.arange(n_wins)
        colors   = ["#2ca02c" if v > VAR_PC1_THRESH else "#ff7f0e" for v in pc1_vals]
        ax3.bar(x_pos, pc1_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
        ax3.axhline(VAR_PC1_THRESH, color="red", linestyle="--", linewidth=2,
                    label=f"Threshold ({VAR_PC1_THRESH})")
        tick_step = 5
        ax3.set_xticks(np.arange(0, n_wins, tick_step))
        ax3.set_xticklabels([str(i) for i in range(0, n_wins, tick_step)])
        ax3.set_ylabel("PC1 Variance Ratio", fontsize=11)
        ax3.set_xlabel("Corrected Window #", fontsize=11)
        ax3.set_title("PC1 Variance in Corrected Windows", fontsize=10, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis="y")

        # ── Panel 4: Removed component (raw − corrected), spans cols 0–1 ──────
        ax4  = fig.add_subplot(gs[1, 0:2])
        diff = df[col1].to_numpy(float) - df_corrected[col1].to_numpy(float)
        ax4.plot(t_zoom, diff[zoom_start:zoom_end], "r-", linewidth=1.5, label="Removed artifact")
        ax4.axvspan(t[i0], t[i1], alpha=0.2, color="red")
        ax4.axhline(0, color="k", linewidth=0.5)
        ax4.set_ylabel("Difference (V)", fontsize=11)
        ax4.set_xlabel("Time (s)", fontsize=11)
        ax4.set_title("Removed Component (Raw - Corrected)", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # ── Panel 5: Full corrected signal (all motion windows shaded) ────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(t, df_corrected[col1].to_numpy(float), "g-", linewidth=0.8)
        for i0_w, i1_w, _ in corrected_windows:
            ax6.axvspan(t[i0_w], t[i1_w], alpha=0.15, color="red")
        ax6.set_ylabel("Intensity (V)", fontsize=10)
        ax6.set_xlabel("Time (s)", fontsize=10)
        ax6.set_title("Full Corrected Signal (all motion windows shaded)", fontsize=10, fontweight="bold")
        ax6.grid(True, alpha=0.3)

        # ── Panel 7: Cardiac visibility PSD (spans full bottom row) ──────────
        ax7 = fig.add_subplot(gs[2, :])
        _plot_cardiac_check(ax7, df_corrected, col1, motion_mask, Fs, t)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"    ⚠️  Plotting failed: {e}")
        import traceback
        traceback.print_exc()


def _plot_cardiac_check(ax, df_corrected, col, motion_mask, Fs, t):
    """
    Find the longest clean segment (outside motion windows), compute its Welch PSD,
    and highlight the cardiac band (~0.7–1.5 Hz) to confirm pulse is preserved.
    """
    sig  = df_corrected[col].to_numpy(float)
    N    = len(sig)
    clean = ~motion_mask

    # Longest contiguous clean segment
    best_start, best_len, i = 0, 0, 0
    while i < N:
        if clean[i]:
            j = i + 1
            while j < N and clean[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start = j - i, i
            i = j
        else:
            i += 1

    seg_start = best_start
    seg_end   = best_start + best_len
    if best_len < int(10 * Fs):   # fallback to whole signal if no clean segment ≥10s
        seg_start, seg_end, best_len = 0, N, N

    clean_seg = sig[seg_start:seg_end]

    nperseg       = min(int(10 * Fs), best_len)
    freqs, psd    = welch(clean_seg, fs=Fs, nperseg=nperseg, noverlap=nperseg // 2)

    ax.semilogy(freqs, psd, color="steelblue", linewidth=1.5, label="PSD (clean segment)")
    ax.axvspan(0.7, 1.5, alpha=0.18, color="orange", label="Cardiac band (0.7–1.5 Hz)")

    # Mark dominant cardiac peak
    cardiac_idx = (freqs >= 0.7) & (freqs <= 1.5)
    if cardiac_idx.any():
        peak_f = freqs[cardiac_idx][np.argmax(psd[cardiac_idx])]
        peak_p = psd[cardiac_idx][np.argmax(psd[cardiac_idx])]
        ax.axvline(peak_f, color="darkorange", linestyle="--", linewidth=2,
                   label=f"Cardiac peak: {peak_f:.2f} Hz")
        ax.scatter([peak_f], [peak_p], color="darkorange", s=60, zorder=5)

    ax.set_xlim(0, min(4.0, Fs / 2))
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("PSD (V²/Hz)", fontsize=11)
    ax.set_title(
        f"Cardiac Visibility Check — PSD of corrected {col.replace('fNIRS ', '')} "
        f"(clean segment: {t[seg_start]:.0f}–{t[seg_end - 1]:.0f} s, "
        f"{best_len / Fs:.0f} s duration)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_all_subjects_summary(results):
    """
    Render a combined summary table (one row per included subject) as a figure.
    """
    rows = []
    for r in results:
        if "included" not in r.get("status", ""):
            continue
        ms = r.get("motion_stats", {})
        di = r.get("debug_info", {})
        cw = di.get("corrected_windows", [])

        n_corrected = ms.get("pca_applied", 0)
        n_skipped   = ms.get("pca_skipped", 0)
        n_detected  = ms.get("n_windows", 0)

        mask_sum  = di.get("motion_mask_sum", 0)
        n_samples = r.get("n_samples", 1)
        pct_str   = f"{100.0 * mask_sum / n_samples:.1f}%" if mask_sum else "—"

        if cw:
            pc1_vals    = [w[2] for w in cw]
            pc1_range   = f"{min(pc1_vals):.3f}–{max(pc1_vals):.3f}"
            largest_str = f"{max(w[1] - w[0] for w in cw) / 20.0:.2f}"
        else:
            pc1_range = largest_str = "—"

        # Use the SAME all-13 ID as the diagnostic figures (e.g. "Subject 07" → "S07").
        # Excluded subjects are skipped above, so their numbers appear as gaps —
        # this keeps every table/figure ID referring to the same recording.
        sid_label = r.get("sid", "—").replace("Subject ", "S")
        rows.append([sid_label, str(n_detected), str(n_corrected), str(n_skipped),
                     pct_str, pc1_range, largest_str])

    if not rows:
        return

    col_labels = ["ID", "Windows\nDetected", "Windows\nCorrected", "Windows\nSkipped",
                  "% Time\nFlagged", "PC1 Variance\nRange", "Largest\nWindow (s)"]

    fig, ax = plt.subplots(figsize=(14, 2.5 + 0.65 * len(rows)))
    ax.axis("off")
    fig.suptitle("Motion Correction Summary — All Subjects", fontsize=15, fontweight="bold")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c5f8a")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    for row_i in range(1, len(rows) + 1):
        bg = "#eef3f8" if row_i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[(row_i, j)].set_facecolor(bg)
            if j == 0:
                tbl[(row_i, j)].set_text_props(fontweight="bold")

    plt.tight_layout()
    plt.show()


# ── Main processing per subject ────────────────────────────────────────────────
def process_subject(subject_folder, subject, plot=False, sid=None):
    """Motion detection + PCA correction for one subject. Returns summary dict + dataframe."""

    in_path = os.path.join(subject_folder, f"{subject}_channel_selected.csv")
    if not os.path.exists(in_path):
        return {"subject": subject, "status": "❌ channel-selected file not found"}, None

    df = pd.read_csv(in_path, low_memory=False)
    Fs = 20.0
    fnirs_cols = [c for c in df.columns if c.startswith("fNIRS")]

    print(f"\n{'='*60}")
    print(f"  {subject}  ({len(df)} samples, {len(df)/Fs:.1f}s)")
    print(f"{'='*60}")

    # ── Step 3: Motion detection and PCA correction ──────────────────────────
    print(f"\n  Motion Detection & Correction:")
    df_original = df.copy()  # Save for motion correction plot
    intensity_corrected, motion_stats, debug_info = detect_and_correct_motion(df, fnirs_cols, Fs=20.0)

    for col in fnirs_cols:
        df[col] = intensity_corrected[col]

    print(f"    ✅ Motion correction complete")

    if plot:
        print(f"    Generating motion correction plot...")
        plot_label = sid if sid else subject
        plot_motion_correction(plot_label, df_original, df, fnirs_cols, Fs=20.0, debug_info=debug_info)

    # ── Save motion-corrected output to subject folder ─────────────────────
    out_path = os.path.join(subject_folder, f"{subject}_motion_corrected.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    return {
        "subject":      subject,
        "sid":          sid,            # shared all-13 ID (same one used on the figures)
        "status":       "✅ included",
        "n_samples":    len(df),
        "n_fnirs_cols": len(fnirs_cols),
        "motion_stats": motion_stats,
        "debug_info":   debug_info,
    }, df


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

    # Build subject-ID map using the same case-sensitive sort as 4.Quality+PPG_results2.py
    # (quality script does sorted(aligned) on filename-derived names — replicated here).
    # Keys are lowercased so folder names (e.g. "Irem") match file names (e.g. "irem").
    aligned_names = []
    for root, _, files in os.walk(DATA_DIR):
        for f in sorted(files):
            if f.endswith("_fnirs_aligned.csv"):
                aligned_names.append(f.replace("_fnirs_aligned.csv", ""))
    aligned_names = sorted(set(aligned_names))          # case-sensitive, mirrors quality script
    sid_map = {name.lower(): f"Subject {i+1:02d}" for i, name in enumerate(aligned_names)}

    print(f"\nFound {len(subject_folders)} subject folder(s)")
    print("Step 5b: Motion Detection & Correction\n")

    results = []

    for i, folder in enumerate(subject_folders):
        subject = os.path.basename(folder)
        # Match bare subject name to the same ID used in the quality-results table
        bare = subject.replace("_custom", "").lower()
        sid  = sid_map.get(bare, f"Subject {i+1:02d}")
        # Plot first 6 subjects for visualization
        plot_this_subject = i < 6
        try:
            result, df = process_subject(folder, subject, plot=plot_this_subject, sid=sid)
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {subject}: {e}")
            results.append({"subject": subject, "status": f"💥 ERROR: {e}"})

    plot_all_subjects_summary(results)


if __name__ == "__main__":
    main()
