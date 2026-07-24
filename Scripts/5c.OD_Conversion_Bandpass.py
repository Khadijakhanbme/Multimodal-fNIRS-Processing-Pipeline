"""
5c_OD_Conversion_Bandpass.py
=============================
Stage 3 of the (split) preprocessing pipeline — OD conversion + bandpass
filtering. Final stage: output filename is unchanged from the original
monolithic 5.Preprocessing.py so downstream scripts (14.GLM_updated_cases.py,
15.hrf_grid_subject.py) keep working without modification.

Input  (per subject, from 5b.Motion_Correction.py):
    {subject}_motion_corrected.csv

Output (per subject):
    {subject}_preprocessed_OD.csv   → consumed by 14.GLM_updated_cases.py,
                                       15.hrf_grid_subject.py

Run:
    python "5c.OD_Conversion_Bandpass.py"
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

# ── OD + bandpass config ──
BP_LO = 0.01                       # bandpass lower cutoff (Hz) — removes slow drift
BP_HI = 0.2                      # bandpass upper cutoff (Hz) — preserves HRF, removes cardiac


# ── Signal processing helpers ────
def bandpass(sig, lo, hi, fs, order=3):
    """Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def convert_to_od_and_filter(df, fnirs_cols, Fs):
    """
    Convert motion-corrected intensity → OD → bandpass-filtered OD.

    I0 per channel = median of the baseline event period.
    Using the pre-task resting baseline (not the full recording) is correct
    because task blocks cause real haemodynamic changes that would bias I0
    if the whole recording were used — same approach as Step3_Preprocessing.py.
    OD = -log(I / I0), clipped to avoid log(0).
    Bandpass: BP_LO–BP_HI Hz (3rd-order Butterworth, zero-phase).

    Returns
    -------
    df_od     : DataFrame with metadata cols + "OD (Xnm) PDY" columns (bandpassed)
    od_pre_bp : dict  col → OD array *before* bandpass  (for diagnostic plot)
    """
    baseline_mask = (df["event_name"] == "baseline").to_numpy(bool)
    n_base = baseline_mask.sum()
    print(f"    I0 reference: baseline period ({n_base} samples, {n_base / Fs:.1f}s)")

    meta_cols = [c for c in df.columns if not c.startswith("fNIRS")]
    df_od     = df[meta_cols].copy()
    od_pre_bp = {}

    for col in fnirs_cols:
        I       = df[col].to_numpy(float)
        I0      = np.median(I[baseline_mask])
        od      = -np.log(np.clip(I, 1e-12, None) / max(float(I0), 1e-12))
        od_filt = bandpass(od, BP_LO, BP_HI, Fs)

        od_col            = col.replace("fNIRS ", "OD ")   # e.g. "OD (740nm) PD1"
        od_pre_bp[od_col] = od
        df_od[od_col]     = od_filt

    return df_od, od_pre_bp


def plot_od_bandpass(subject, df_od, od_pre_bp, Fs):
    """
    Log-log PSD comparison (OD before vs after bandpass) for the first 740nm
    and 850nm long-sep channels.  Confirms the passband and shows what is removed.
    """
    try:
        od_cols   = [c for c in df_od.columns if c.startswith("OD")]
        cols_740  = [c for c in od_cols if "740nm" in c]
        cols_850  = [c for c in od_cols if "850nm" in c]
        rep_cols  = ([cols_740[0]] if cols_740 else []) + ([cols_850[0]] if cols_850 else [])

        if not rep_cols:
            return

        N       = len(df_od)
        nperseg = min(int(20 * Fs), N // 2)

        fig, axes = plt.subplots(1, len(rep_cols), figsize=(6 * len(rep_cols), 4.5),
                                 sharey=True)
        if len(rep_cols) == 1:
            axes = [axes]

        fig.suptitle(f"{subject} — OD Conversion & Bandpass ({BP_LO}–{BP_HI} Hz)",
                     fontsize=14, fontweight="bold")

        for ax, col in zip(axes, rep_cols):
            f_pre,  pxx_pre  = welch(od_pre_bp[col], fs=Fs,
                                     nperseg=nperseg, noverlap=nperseg // 2)
            f_post, pxx_post = welch(df_od[col].to_numpy(float), fs=Fs,
                                     nperseg=nperseg, noverlap=nperseg // 2)

            fmask = f_pre > 0
            ax.loglog(f_pre[fmask],  pxx_pre[fmask],  color="#aaaaaa", lw=1.2,
                      label="OD (before bandpass)")
            ax.loglog(f_post[fmask], pxx_post[fmask], color="#1f77b4", lw=1.8,
                      label="OD (after bandpass)")
            ax.axvspan(BP_LO, BP_HI, color="#b0d4f1", alpha=0.35,
                       label=f"Passband {BP_LO}–{BP_HI} Hz")
            ax.axvline(BP_LO, color="steelblue", lw=1.0, linestyle=":")
            ax.axvline(BP_HI, color="steelblue", lw=1.0, linestyle=":")

            ax.set_xlim(f_pre[fmask][0], min(4.0, Fs / 2))
            ax.set_xlabel("Frequency (Hz)", fontsize=11)
            ax.set_ylabel("PSD (OD²/Hz)", fontsize=11)
            ax.set_title(col.replace("OD ", ""), fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, which="both", alpha=0.25)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"    ⚠️  OD bandpass plot failed: {e}")
        import traceback
        traceback.print_exc()


def plot_intensity_to_od(subject, df, od_pre_bp, t, pd_name="PD1"):
    """
    2×2 time-domain plot showing the intensity → OD conversion step.
    Top row: raw intensity (I) at 740 nm and 850 nm.
    Bottom row: OD = -log(I/I0) at 740 nm and 850 nm (before bandpass).
    """
    try:
        cols_I  = [f"fNIRS (740nm) {pd_name}", f"fNIRS (850nm) {pd_name}"]
        cols_OD = [f"OD (740nm) {pd_name}",    f"OD (850nm) {pd_name}"]
        wl_labels = ["740 nm", "850 nm"]
        colors    = ["#4e79a7", "#e15759"]

        if not all(c in df.columns for c in cols_I):
            # Fallback: try PD2
            pd_name = "PD2"
            cols_I  = [f"fNIRS (740nm) {pd_name}", f"fNIRS (850nm) {pd_name}"]
            cols_OD = [f"OD (740nm) {pd_name}",    f"OD (850nm) {pd_name}"]
        if not all(c in df.columns for c in cols_I):
            return

        n  = min(len(t), len(df))
        tp = t[:n]

        fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)

        for j, (col_I, col_OD, wl, color) in enumerate(
                zip(cols_I, cols_OD, wl_labels, colors)):
            # Top: raw intensity
            ax_top = axes[0, j]
            I = df[col_I].to_numpy(float)[:n]
            ax_top.plot(tp, I, color=color, lw=0.8)
            ax_top.set_ylabel("Intensity (V)", fontsize=10)
            ax_top.set_title(f"{wl} — Raw Intensity", fontsize=10, fontweight="bold")
            ax_top.grid(alpha=0.3)

            # Bottom: OD
            ax_bot = axes[1, j]
            if col_OD in od_pre_bp:
                OD = od_pre_bp[col_OD][:n]
                ax_bot.plot(tp, OD, color=color, lw=0.8)
            ax_bot.set_ylabel("ΔOD (a.u.)", fontsize=10)
            ax_bot.set_title(f"{wl} — After OD Conversion  [−log(I/I₀)]",
                             fontsize=10, fontweight="bold")
            ax_bot.set_xlabel("Time (s)", fontsize=10)
            ax_bot.axhline(0, color="k", lw=0.4)
            ax_bot.grid(alpha=0.3)

        fig.suptitle(f"{subject}  —  {pd_name}  |  Intensity → OD conversion\n"
                     f"Top: raw detector voltage  |  Bottom: optical density [−log(I/I₀)]",
                     fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    except Exception as e:
        print(f"    ⚠️  Intensity→OD plot failed: {e}")


def plot_bandpass_timedomain_zoom(subject, df_od, od_pre_bp, zoom_s=60.0, pd_name="PD1"):
    """
    Time-domain zoomed comparison: OD before vs after bandpass filtering.
    Zoom window is centred on the middle of the recording so a task block
    is likely visible. Shows 740 nm and 850 nm side by side.
    """
    try:
        t = df_od["time_s"].to_numpy(float) if "time_s" in df_od.columns \
            else np.arange(len(df_od)) / 20.0

        # Try PD1, fall back to PD2
        for pd in (pd_name, "PD2", "PD3"):
            cols_pre  = [f"OD (740nm) {pd}", f"OD (850nm) {pd}"]
            cols_post = [c for c in cols_pre if c in df_od.columns]
            if len(cols_post) == 2 and all(c in od_pre_bp for c in cols_pre):
                pd_name = pd
                break
        else:
            return

        # Centre zoom on the middle third of the recording (likely task blocks)
        t_mid   = t[len(t) // 2]
        t_start = max(t[0],  t_mid - zoom_s / 2)
        t_end   = min(t[-1], t_mid + zoom_s / 2)
        mask    = (t >= t_start) & (t <= t_end)
        tp      = t[mask]

        wl_list    = ["740nm", "850nm"]
        colors     = ["#4e79a7", "#e15759"]
        wl_labels  = ["740 nm", "850 nm"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=False)

        for j, (wl, color, wl_lbl) in enumerate(zip(wl_list, colors, wl_labels)):
            col = f"OD ({wl}) {pd_name}"
            pre  = od_pre_bp[col][mask]
            post = df_od[col].to_numpy(float)[mask]

            ax_pre  = axes[0, j]
            ax_post = axes[1, j]

            ax_pre.plot(tp, pre, color=color, lw=0.9, alpha=0.9)
            ax_pre.set_title(f"{wl_lbl} — Before bandpass", fontsize=10, fontweight="bold")
            ax_pre.set_ylabel("ΔOD (a.u.)", fontsize=10)
            ax_pre.axhline(0, color="k", lw=0.4)
            ax_pre.grid(alpha=0.3)

            ax_post.plot(tp, post, color=color, lw=0.9, alpha=0.9)
            ax_post.set_title(f"{wl_lbl} — After bandpass ({BP_LO}–{BP_HI} Hz)",
                              fontsize=10, fontweight="bold")
            ax_post.set_ylabel("ΔOD (a.u.)", fontsize=10)
            ax_post.set_xlabel("Time (s)", fontsize=10)
            ax_post.axhline(0, color="k", lw=0.4)
            ax_post.grid(alpha=0.3)

        fig.suptitle(
            f"{subject}  —  {pd_name}  |  Bandpass filtering  ({BP_LO}–{BP_HI} Hz)\n"
            f"Zoomed window: {t_start:.0f}–{t_end:.0f} s  "
            f"(centre of recording — likely contains task blocks)",
            fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    except Exception as e:
        print(f"    ⚠️  Bandpass zoom plot failed: {e}")


# ── Main processing per subject ────────────────────────────────────────────────
def process_subject(subject_folder, subject, plot=False, sid=None):
    """OD conversion + bandpass filtering for one subject. Returns summary dict + dataframe."""

    in_path = os.path.join(subject_folder, f"{subject}_motion_corrected.csv")
    if not os.path.exists(in_path):
        return {"subject": subject, "status": "❌ motion-corrected file not found"}, None

    df = pd.read_csv(in_path, low_memory=False)
    Fs = 20.0
    fnirs_cols = [c for c in df.columns if c.startswith("fNIRS")]

    print(f"\n{'='*60}")
    print(f"  {subject}  ({len(df)} samples, {len(df)/Fs:.1f}s)")
    print(f"{'='*60}")

    plot_label = sid if sid else subject

    # ── Step 4: OD conversion + bandpass filtering ──────────────────────────
    print(f"\n  OD Conversion & Bandpass Filtering ({BP_LO}–{BP_HI} Hz):")
    df_od, od_pre_bp = convert_to_od_and_filter(df, fnirs_cols, Fs=20.0)
    od_cols = [c for c in df_od.columns if c.startswith("OD")]

    for col in od_cols:
        sig = df_od[col].to_numpy(float)
        print(f"    {col:<30}  mean={sig.mean():.4f}  std={sig.std():.4f}")

    print(f"    ✅ OD conversion + bandpass complete ({len(od_cols)} channels)")

    if plot:
        print(f"    Generating Intensity → OD conversion plot...")
        t_arr = df_od["time_s"].to_numpy(float) if "time_s" in df_od.columns \
                else np.arange(len(df_od)) / 20.0
        plot_intensity_to_od(plot_label, df, od_pre_bp, t_arr)
        print(f"    Generating bandpass zoom (time-domain) plot...")
        plot_bandpass_timedomain_zoom(plot_label, df_od, od_pre_bp)
        print(f"    Generating OD bandpass diagnostic plot...")
        plot_od_bandpass(plot_label, df_od, od_pre_bp, Fs=20.0)

    # ── Save preprocessed output to subject folder ───────────────────────────
    out_path = os.path.join(subject_folder, f"{subject}_preprocessed_OD.csv")
    df_od.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    return {
        "subject":      subject,
        "sid":          sid,
        "status":       "✅ included",
        "n_samples":    len(df_od),
        "n_od_cols":    len(od_cols),
    }, df_od


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(results):
    print(f"\n{'='*70}")
    print(f"  OD CONVERSION & BANDPASS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Subject':<12} {'Include':>8} {'OD channels':>12} {'Samples':>8}")
    print(f"  {'-'*55}")
    for r in results:
        included = "✅" if "included" in r["status"] else "❌"
        if "included" not in r["status"]:
            print(f"  {r['subject']:<12} {included:>8}  {r['status']}")
        else:
            print(f"  {r['subject']:<12} {included:>8} {r.get('n_od_cols', 0):>12} "
                  f"{r.get('n_samples', 0):>8}")
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

    # Build subject-ID map using the same case-sensitive sort as 4.Quality+PPG_results2.py
    aligned_names = []
    for root, _, files in os.walk(DATA_DIR):
        for f in sorted(files):
            if f.endswith("_fnirs_aligned.csv"):
                aligned_names.append(f.replace("_fnirs_aligned.csv", ""))
    aligned_names = sorted(set(aligned_names))          # case-sensitive, mirrors quality script
    sid_map = {name.lower(): f"Subject {i+1:02d}" for i, name in enumerate(aligned_names)}

    print(f"\nFound {len(subject_folders)} subject folder(s)")
    print("Step 5c: OD Conversion & Bandpass Filtering\n")

    results = []

    for i, folder in enumerate(subject_folders):
        subject = os.path.basename(folder)
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

    print_summary(results)


if __name__ == "__main__":
    main()
