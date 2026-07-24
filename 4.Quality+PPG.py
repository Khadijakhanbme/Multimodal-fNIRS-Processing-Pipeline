import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr

# ---------------- Config ----------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_SCRIPT_DIR, "..", "rawdata", "2nd Trial", "Multimodal_data")

WLS          = (740, 770, 810, 850)
PHOTODIODES  = (0, 1, 2, 3)
PPG_WL_RED   = 635

SNR_THRESH   = 1.5            # linear — Homer3 standard (mean/std ratio)
SCI_THRESH   = 0.7            # reported only, not used for pass/fail
HR_BAND      = (1.0, 2.5)     # Hz — excludes Mayer waves (0.5-1.0 Hz)
BPM_TOL      = 6.0            # BPM — PPG match tolerance
PROMINENCE   = 2.0            # cardiac peak must be >2x local mean
MIN_PPG_SEG  = 60.0           # s — minimum stable PPG segment for HR detection

# ── Coupling detection config (moved from Preprocessing) ──
COUPLING_THRESH = 0.05        # V — step change threshold for shift detection
DETECT_WL       = 810         # wavelength used for shift detection (near isosbestic)


# ---------------- Signal helpers ----------------
def bandpass(sig, lo, hi, fs, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def welch_peak(sig, fs, band):
    f, pxx = welch(sig, fs=fs, nperseg=512, noverlap=256, detrend="linear")
    mask = (f >= band[0]) & (f <= band[1])
    peak_hz = f[mask][np.argmax(pxx[mask])]
    return peak_hz, f, pxx


def sci_segment(df, pd_idx, Fs, hr_band, min_seg_s=20):
    """Compute SCI on longest continuous valid segment."""
    valid = (df["gap_mask"] == 1).to_numpy() if "gap_mask" in df.columns \
            else np.ones(len(df), dtype=bool)

    best_start, best_len = 0, 0
    cur_start, cur_len   = 0, 0
    for i in range(len(valid)):
        if valid[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len  = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    seg_dur = best_len / Fs
    if best_len < int(min_seg_s * Fs):
        return np.nan, seg_dur

    s740 = df[f"fNIRS (740nm) PD{pd_idx}"].iloc[best_start:best_start+best_len].to_numpy(float)
    s850 = df[f"fNIRS (850nm) PD{pd_idx}"].iloc[best_start:best_start+best_len].to_numpy(float)
    f740 = bandpass(s740, hr_band[0], hr_band[1], Fs)
    f850 = bandpass(s850, hr_band[0], hr_band[1], Fs)
    sci  = abs(pearsonr(f740, f850)[0])
    return sci, seg_dur


def get_ppg_signal(df, Fs):
    """
    Return PPG signal for HR detection, handling TIA gain switches.
    - No switch: use full signal
    - Switch found: use longer stable half if >= MIN_PPG_SEG seconds
    - Switch found but no stable half >= MIN_PPG_SEG: return None (unreliable)
    Returns: (ppg_sig, ppg_status_string)
    """
    ppg_raw = df[f"PPG ({PPG_WL_RED}nm)"].to_numpy(float)
    tia_col = f"TIA gain ({PPG_WL_RED}nm)"

    if tia_col not in df.columns:
        return ppg_raw, "clean"

    tia = df[tia_col].to_numpy(float)
    switch_idx = np.where(np.diff(tia) != 0)[0]

    if len(switch_idx) == 0:
        return ppg_raw, "clean"

    # Switch detected — find stable halves
    split  = switch_idx[-1] + 10
    pre    = ppg_raw[:switch_idx[0]]
    post   = ppg_raw[split:]
    pre_ok  = len(pre)  / Fs >= MIN_PPG_SEG
    post_ok = len(post) / Fs >= MIN_PPG_SEG

    if post_ok and len(post) >= len(pre):
        return post, f"TIA switch — post-switch segment ({len(post)/Fs:.0f}s)"
    elif pre_ok:
        return pre,  f"TIA switch — pre-switch segment ({len(pre)/Fs:.0f}s)"
    elif post_ok:
        return post, f"TIA switch — post-switch segment ({len(post)/Fs:.0f}s)"
    else:
        return None, "TIA switch — no stable segment ≥60s — PPG UNRELIABLE"


# ── Coupling shift detection (moved from Preprocessing) ─────────────────────
def detect_all_shifts(df, pd_idx):
    """
    Detect ALL coupling shifts using the 810nm signal (near isosbestic point).
    Looks for abrupt step changes above COUPLING_THRESH.
    Only checks within experiment time window (gap_mask == 1).
    Excludes shifts within 30 seconds of experiment start/end to avoid edge artifacts.
    Returns sorted list of shift sample indices (empty if none found).
    """
    col_810 = f"fNIRS ({DETECT_WL}nm) PD{pd_idx}"
    if col_810 not in df.columns:
        return []
    
    # Respect gap_mask to exclude pre/post-experiment samples
    valid = (df["gap_mask"] == 1).to_numpy() if "gap_mask" in df.columns \
            else np.ones(len(df), dtype=bool)
    
    # Find valid experiment window boundaries
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return []
    
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    
    # Estimate Fs if not available - use median diff from timestamps
    if "timestamp" in df.columns:
        t_raw = pd.to_datetime(df["timestamp"], errors="coerce")
        t = (t_raw - t_raw.iloc[0]).dt.total_seconds().to_numpy(float)
        fs_local = 1.0 / np.median(np.diff(t))
    else:
        fs_local = 100.0  # fallback
    
    # Create buffer zone: exclude shifts within 30 seconds from experiment boundaries
    buffer_samples = int(30 * fs_local)
    start_margin = first_valid + buffer_samples
    end_margin = last_valid - buffer_samples
    
    sig = df[col_810].to_numpy(float)
    steps = np.abs(np.diff(sig))
    
    # Find all steps above threshold, but only within valid experiment window AND away from edges
    raw_shifts = []
    for i in range(len(steps) - 1):
        if (steps[i] > COUPLING_THRESH and valid[i] and valid[i + 1] 
            and i >= start_margin and i <= end_margin):
            raw_shifts.append(i)
    
    if not raw_shifts:
        return []
    # Cluster: merge shifts within 50 samples of each other → keep first of each cluster
    clustered = [raw_shifts[0]]
    for s in raw_shifts[1:]:
        if s - clustered[-1] > 50:
            clustered.append(s)
    return clustered


def fmt_num(v, decimals=2):
    if isinstance(v, float) and np.isnan(v):
        return "--"
    return f"{v:.{decimals}f}"


def fmt_bool(v):
    if isinstance(v, float) and np.isnan(v):
        return "--"
    return "True" if v else "False"


# ---------------- Per-subject processing ----------------
def process_subject(data_path, participant):
    """Run full quality assessment for one subject. Returns summary dict."""

    df = pd.read_csv(data_path)
    t_raw = pd.to_datetime(df["timestamp"], errors="coerce")
    t  = (t_raw - t_raw.iloc[0]).dt.total_seconds().to_numpy(float)
    N  = len(df)
    Fs = 1.0 / np.median(np.diff(t))

    print(f"\n{'='*60}")
    print(f"  {participant}  ({N} samples, Fs={Fs:.2f}Hz)")
    print(f"{'='*60}")

    # ---- PPG HR — with TIA switch handling ----
    ppg_sig, ppg_status = get_ppg_signal(df, Fs)
    print(f"  PPG: {ppg_status}")

    ppg_reliable = ppg_sig is not None
    if ppg_reliable:
        ppg_hr_hz, ppg_f, ppg_pxx = welch_peak(ppg_sig, Fs, HR_BAND)
        ppg_hr_bpm = 60.0 * ppg_hr_hz
        print(f"  PPG HR: {ppg_hr_bpm:.1f} BPM")
    else:
        ppg_hr_hz  = np.nan
        ppg_hr_bpm = np.nan
        ppg_f      = np.array([])
        ppg_pxx    = np.array([])
        print(f"  ⚠️  PPG HR cannot be determined — all channels will fail PPG match")

    # ---- IMU sanity check ----
    imu_status = "OK"
    imu_cols = ["IMU (ax)", "IMU (ay)", "IMU (az)"]

    if all(col in df.columns for col in imu_cols):
        ax = df["IMU (ax)"].to_numpy(dtype=float)
        ay = df["IMU (ay)"].to_numpy(dtype=float)
        az = df["IMU (az)"].to_numpy(dtype=float)

        imu_flags = []

        for name, sig in [("ax", ax), ("ay", ay), ("az", az)]:
            nan_pct = 100 * np.isnan(sig).sum() / len(sig)
            if nan_pct > 5:
                imu_flags.append(f"{name}: >5% NaNs")

            if np.nanstd(sig) < 0.001:
                imu_flags.append(f"{name}: frozen")

        if imu_flags:
            imu_status = "WARNING: " + "; ".join(imu_flags)

    print(f"  IMU: {imu_status}")

    # ---- SCI per photodiode ----
    sci_per_pd     = {}
    sci_dur_per_pd = {}
    for pd_idx in PHOTODIODES:
        sci, seg_dur = sci_segment(df, pd_idx, Fs, HR_BAND)
        sci_per_pd[pd_idx]     = sci
        sci_dur_per_pd[pd_idx] = seg_dur

    # ---- PPG match per PD (740nm and 850nm) ----
    ppg_match_740 = {}
    ppg_match_850 = {}
    bpm_740       = {}
    bpm_850       = {}

    for pd_idx in PHOTODIODES:
        # 740 nm
        sig_740 = df[f"fNIRS (740nm) PD{pd_idx}"].to_numpy(float)
        peak_740, f_740, pxx_740 = welch_peak(sig_740, Fs, HR_BAND)
        bpm_740[pd_idx] = 60.0 * peak_740
        hr_mask_740     = (f_740 >= HR_BAND[0]) & (f_740 <= HR_BAND[1])
        prom_740        = np.max(pxx_740[hr_mask_740]) / np.mean(pxx_740[hr_mask_740])
        ppg_match_740[pd_idx] = bool(
            ppg_reliable
            and abs(bpm_740[pd_idx] - ppg_hr_bpm) <= BPM_TOL
            and prom_740 > PROMINENCE
        )

        # 850 nm
        sig_850 = df[f"fNIRS (850nm) PD{pd_idx}"].to_numpy(float)
        peak_850, f_850, pxx_850 = welch_peak(sig_850, Fs, HR_BAND)
        bpm_850[pd_idx] = 60.0 * peak_850
        hr_mask_850     = (f_850 >= HR_BAND[0]) & (f_850 <= HR_BAND[1])
        prom_850        = np.max(pxx_850[hr_mask_850]) / np.mean(pxx_850[hr_mask_850])
        ppg_match_850[pd_idx] = bool(
            ppg_reliable
            and abs(bpm_850[pd_idx] - ppg_hr_bpm) <= BPM_TOL
            and prom_850 > PROMINENCE
        )

    # ---- Build metrics table (without coupling — that comes after pass/fail) ----
    rows = []
    for wl in WLS:
        for pd_idx in PHOTODIODES:
            col    = f"fNIRS ({wl}nm) PD{pd_idx}"
            sig    = df[col].to_numpy(float)
            snr_linear = np.mean(sig) / np.std(sig)

            snr_740 = np.mean(df[f"fNIRS (740nm) PD{pd_idx}"].to_numpy(float)) / np.std( df[f"fNIRS (740nm) PD{pd_idx}"].to_numpy(float))
            
            snr_850 = np.mean(df[f"fNIRS (850nm) PD{pd_idx}"].to_numpy(float)) /np.std( df[f"fNIRS (850nm) PD{pd_idx}"].to_numpy(float)
            )

            pass_q = bool(
                (snr_740 >= SNR_THRESH) and
                (snr_850 >= SNR_THRESH) and
                ppg_match_740[pd_idx]   and
                ppg_match_850[pd_idx]
            )

            rows.append({
                "wl":            wl,
                "pd":            pd_idx,
                "SNR_linear":    snr_linear,
                "SNR_740nm":     snr_740,
                "SNR_850nm":     snr_850,
                "SCI":           sci_per_pd[pd_idx],
                "SCI_seg_dur_s": sci_dur_per_pd[pd_idx],
                "fnirs_bpm_740": bpm_740[pd_idx],
                "fnirs_bpm_850": bpm_850[pd_idx],
                "ppg_match_740": ppg_match_740[pd_idx],
                "ppg_match_850": ppg_match_850[pd_idx],
                "pass_quality":  pass_q,
            })

    metrics = pd.DataFrame(rows)

    # ---- Identify passing PDs ----
    passing_pds = [
        pd_idx for pd_idx in PHOTODIODES
        if metrics[(metrics["wl"]==740) & (metrics["pd"]==pd_idx)].iloc[0]["pass_quality"]
    ]
    passing_long_sep = [p for p in passing_pds if p > 0]   # PD1/2/3 only

    # ---- Coupling shift detection — on passing long-sep PDs AND PD0 (regardless of pass/fail) ----
    coupling_per_pd = {}   # pd_idx → list of shift sample indices
    pds_to_check = passing_long_sep + [0]  # Always check PD0, plus passing long-sep PDs
    pds_to_check = sorted(list(set(pds_to_check)))  # Remove duplicates and sort
    
    print(f"\n  Coupling check (passing long-sep PDs: {passing_long_sep}; also checking PD0):")
    for pd_idx in pds_to_check:
        shifts = detect_all_shifts(df, pd_idx)
        coupling_per_pd[pd_idx] = shifts
        pd_label = "PD0 (short-sep)" if pd_idx == 0 else f"PD{pd_idx}"
        if shifts:
            positions = ", ".join([f"{s/Fs:.1f}s" for s in shifts])
            print(f"    {pd_label}: ⚠️  {len(shifts)} shift(s) at [{positions}]")
        else:
            print(f"    {pd_label}: ✅ no coupling shift")

    # Add coupling column to metrics (NaN for PDs not checked)
    metrics["n_coupling_shifts"] = metrics["pd"].apply(
        lambda p: len(coupling_per_pd[p]) if p in coupling_per_pd else np.nan
    )

    # ---- Print per-PD summary ----
    print(f"\n  {'PD':>3}  {'SNR_740':>8}  {'SNR_850':>8}  "
          f"{'BPM_740':>8}  {'BPM_850':>8}  "
          f"{'PPG_740':>8}  {'PPG_850':>8}  {'SCI':>6}  {'Shifts':>7}  {'PASS':>6}")
    print("  " + "-"*90)
    for pd_idx in PHOTODIODES:
        r = metrics[(metrics["wl"]==740) & (metrics["pd"]==pd_idx)].iloc[0]
        passed = r["pass_quality"]
        # Coupling: show count for checked PDs, "--" for unchecked (failed or PD0)
        if pd_idx in coupling_per_pd:
            n_s = len(coupling_per_pd[pd_idx])
            shift_disp = f"⚠️ {n_s}" if n_s > 0 else "✅  0"
        else:
            shift_disp = "  --"
        print(
            f"  {pd_idx:>3}  "
            f"{fmt_num(r['SNR_740nm']):>8}  "
            f"{fmt_num(r['SNR_850nm']):>8}  "
            f"{fmt_num(r['fnirs_bpm_740']):>8}  "
            f"{fmt_num(r['fnirs_bpm_850']):>8}  "
            f"{fmt_bool(r['ppg_match_740']):>8}  "
            f"{fmt_bool(r['ppg_match_850']):>8}  "
            f"{fmt_num(sci_per_pd[pd_idx]):>6}  "
            f"{shift_disp:>7}  "
            f"{'✓ PASS' if passed else '✗ FAIL':>6}"
        )

    # ---- Save CSV ----
    out_csv = os.path.join(os.path.dirname(data_path),
                           f"{participant}_quality_metrics.csv")
    metrics.to_csv(out_csv, index=False)

    # ---- Figures ----
    _plot_quality(metrics, sci_per_pd, bpm_740, bpm_850,
                  ppg_hr_bpm, ppg_hr_hz, ppg_f, ppg_pxx,
                  df, Fs, participant)

    # ---- Coupling time-domain plot (only for checked PDs that have shifts) ----
    if any(len(v) > 0 for v in coupling_per_pd.values()):
        _plot_coupling(df, coupling_per_pd, Fs, participant)

    # ---- Return summary for cross-subject table ----
    total_shifts = sum(len(v) for v in coupling_per_pd.values())
    pds_with_shifts = [p for p, v in coupling_per_pd.items() if len(v) > 0]
    pd0_shifts = len(coupling_per_pd.get(0, []))

    return {
        "subject":          participant,
        "ppg_hr":           ppg_hr_bpm,
        "ppg_status":       ppg_status,
        "imu_status":       imu_status,
        "PD0":              0 in passing_pds,
        "PD1":              1 in passing_pds,
        "PD2":              2 in passing_pds,
        "PD3":              3 in passing_pds,
        "n_pass":           len(passing_pds),
        "passing_pds":      passing_pds,
        "total_shifts":     total_shifts,
        "pds_with_shifts":  pds_with_shifts,
        "pd0_shifts":       pd0_shifts,
    }


# ---------------- Plots ----------------
def _plot_quality(metrics, sci_per_pd, bpm_740, bpm_850,
                  ppg_hr_bpm, ppg_hr_hz, ppg_f, ppg_pxx,
                  df, Fs, participant):

    wl_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    bar_w     = 0.18
    x         = np.arange(len(PHOTODIODES))
    pd_labels = [f"PD{i}" for i in PHOTODIODES]

    # Figure 1 — SNR / SCI / PPG match bars
    fig1, (ax_snr, ax_sci, ax_ppg) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for wi, wl in enumerate(WLS):
        snr_vals = [metrics[(metrics["wl"]==wl) & (metrics["pd"]==p)].iloc[0]["SNR_linear"]
                    for p in PHOTODIODES]
        sci_vals = [metrics[(metrics["wl"]==wl) & (metrics["pd"]==p)].iloc[0]["SCI"]
                    for p in PHOTODIODES]
        offset   = (wi - 1.5) * bar_w
        ax_snr.bar(x + offset, snr_vals, bar_w, label=f"{wl} nm",
                   color=wl_colors[wi], alpha=0.85, edgecolor="white")
        ax_sci.bar(x + offset, sci_vals, bar_w,
                   color=wl_colors[wi], alpha=0.85, edgecolor="white")

    delta_740 = [abs(bpm_740[p] - ppg_hr_bpm) if not np.isnan(ppg_hr_bpm) else 0
                 for p in PHOTODIODES]
    delta_850 = [abs(bpm_850[p] - ppg_hr_bpm) if not np.isnan(ppg_hr_bpm) else 0
                 for p in PHOTODIODES]
    ax_ppg.bar(x - 0.2, delta_740, 0.35, label="Δ BPM (740nm)",
               color="#4e79a7", alpha=0.85, edgecolor="white")
    ax_ppg.bar(x + 0.2, delta_850, 0.35, label="Δ BPM (850nm)",
               color="#e15759", alpha=0.85, edgecolor="white")
    ax_ppg.axhline(BPM_TOL, color="red", lw=1.5, ls="--",
                   label=f"Tolerance (±{BPM_TOL} BPM)")
    ax_ppg.set_ylabel("Δ BPM from PPG", fontsize=11)
    ax_ppg.set_ylim(0, max(max(delta_740), max(delta_850), 1) * 1.3 + 2)
    ax_ppg.legend(fontsize=9, loc="upper right")
    ax_ppg.grid(axis="y", alpha=0.3)

    ax_snr.axhline(SNR_THRESH, color="red", lw=1.5, ls="--",
                   label=f"Threshold ({SNR_THRESH} linear)")
    ax_snr.set_ylabel("SNR (linear)", fontsize=11)
    ax_snr.set_ylim(0, max(metrics["SNR_linear"]) * 1.15)
    ax_snr.legend(fontsize=9, loc="upper right", ncol=5)
    ax_snr.grid(axis="y", alpha=0.3)
    ax_snr.set_title("Channel Quality — SNR, SCI (informational), PPG Cardiac Match",
                     fontsize=12)

    ax_sci.axhline(SCI_THRESH, color="gray", lw=1.5, ls="--",
                   label=f"Literature threshold ({SCI_THRESH}) — informational")
    ax_sci.set_ylabel("SCI (informational)", fontsize=11)
    ax_sci.set_ylim(0, 1.15)
    ax_sci.legend(fontsize=9, loc="upper right")
    ax_sci.grid(axis="y", alpha=0.3)

    for p in PHOTODIODES:
        passed = metrics[(metrics["wl"]==740) & (metrics["pd"]==p)].iloc[0]["pass_quality"]
        if not passed:
            for ax in [ax_snr, ax_sci, ax_ppg]:
                ax.axvspan(p - 0.45, p + 0.45, color="red", alpha=0.06, zorder=0)

    ax_ppg.set_xticks(x)
    ax_ppg.set_xticklabels(pd_labels, fontsize=11)
    ax_ppg.set_xlabel("Photodiode (source-detector separation)", fontsize=11)
    for p in PHOTODIODES:
        passed = metrics[(metrics["wl"]==740) & (metrics["pd"]==p)].iloc[0]["pass_quality"]
        label  = "✓ Pass" if passed else "✗ Fail"
        color  = "#2ca02c" if passed else "#d62728"
        ax_ppg.text(p, -0.22, label, ha="center", va="top",
                    fontsize=10, color=color, fontweight="bold",
                    transform=ax_ppg.get_xaxis_transform())

    fig1.suptitle(
        f"Channel Quality Summary — Subject: {participant}\n"
        f"Pass criterion: SNR≥{SNR_THRESH} linear (740+850nm) AND PPG cardiac match ±{BPM_TOL}BPM\n"
        f"SCI shown for reference only",
        fontsize=11
    )
    fig1.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Figure 2 — PSD per PD
    fig2, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
    axes = axes.flatten()

    for pd_idx in PHOTODIODES:
        ax     = axes[pd_idx]
        passed = metrics[(metrics["wl"]==740) & (metrics["pd"]==pd_idx)].iloc[0]["pass_quality"]

        if len(ppg_f) > 0:
            ax.semilogy(ppg_f, ppg_pxx, color="#d62728", lw=2.0,
                        label=f"PPG — {ppg_hr_bpm:.1f} BPM")
            if not np.isnan(ppg_hr_hz):
                ax.axvline(ppg_hr_hz, color="#d62728", ls="--", lw=1.2)

        sig_740 = df[f"fNIRS (740nm) PD{pd_idx}"].to_numpy(float)
        f_740, pxx_740 = welch(sig_740, fs=Fs, nperseg=512, noverlap=256, detrend="linear")
        ax.semilogy(f_740, pxx_740, color="#4e79a7", lw=1.5, alpha=0.8,
                    label=f"740nm — {bpm_740[pd_idx]:.1f} BPM")

        sig_850 = df[f"fNIRS (850nm) PD{pd_idx}"].to_numpy(float)
        f_850, pxx_850 = welch(sig_850, fs=Fs, nperseg=512, noverlap=256, detrend="linear")
        ax.semilogy(f_850, pxx_850, color="#e15759", lw=1.5, alpha=0.8,
                    label=f"850nm — {bpm_850[pd_idx]:.1f} BPM")

        ax.axvspan(HR_BAND[0], HR_BAND[1], color="#cccccc", alpha=0.3)
        ax.set_xlim(0, 3)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (a.u.²/Hz)")
        status = "✓ PASS" if passed else "✗ FAIL"
        color  = "#2ca02c" if passed else "#d62728"
        ax.set_title(f"PD{pd_idx} — {status}  |  SCI={fmt_num(sci_per_pd[pd_idx])}",
                     color=color, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    fig2.suptitle(
        f"PPG vs fNIRS Cardiac PSD — Subject: {participant}\n"
        f"740nm (blue) and 850nm (red) shown for all PDs",
        fontsize=12
    )
    fig2.tight_layout()
    plt.show()


# ── Coupling shift time-domain plot (new) ────────────────────────────────────
def _plot_coupling(df, coupling_per_pd, Fs, participant):
    """
    Time-domain plot of 810nm signal for each checked PD (passing long-sep),
    with red vertical lines marking detected coupling shifts.
    Only called when at least one checked PD has a shift.
    """
    checked_pds = sorted(coupling_per_pd.keys())
    n_pds = len(checked_pds)
    if n_pds == 0:
        return

    fig, axes = plt.subplots(n_pds, 1, figsize=(14, 2.5 * n_pds), sharex=True)
    if n_pds == 1:
        axes = [axes]

    t = np.arange(len(df)) / Fs

    for ax, pd_idx in zip(axes, checked_pds):
        col = f"fNIRS ({DETECT_WL}nm) PD{pd_idx}"
        if col not in df.columns:
            ax.set_visible(False)
            continue

        sig = df[col].to_numpy(float)
        ax.plot(t, sig, color="#1f77b4", lw=0.6, alpha=0.9)

        shifts = coupling_per_pd[pd_idx]
        for i, s in enumerate(shifts):
            label = "coupling shift" if i == 0 else None
            ax.axvline(s / Fs, color="red", ls="--", lw=1.5, label=label)

        n = len(shifts)
        tag = f"⚠️ {n} shift(s)" if n > 0 else "✅ clean"
        ax.set_ylabel(f"PD{pd_idx}\n{DETECT_WL}nm (V)", fontsize=9)
        ax.set_title(f"PD{pd_idx} — {tag}", fontsize=10,
                     color="#d62728" if n > 0 else "#2ca02c", fontweight="bold")
        ax.grid(True, alpha=0.25)
        if shifts:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(
        f"Coupling Shift Detection ({DETECT_WL}nm) — Subject: {participant}\n"
        f"Threshold: {COUPLING_THRESH} V step change  |  Checked PDs: {checked_pds}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------------- Cross-subject summary table ----------------
def print_cross_subject_summary(summaries):
    W = 112
    print(f"\n{'='*W}")
    print(f"  CHANNEL QUALITY SUMMARY — ALL SUBJECTS")
    print(f"{'='*W}")
    print(f"  {'ID':<5} {'PPG_HR':>7}  "
          f"{'PD0':>5} {'PD1':>5} {'PD2':>5} {'PD3':>5}  "
          f"{'Pass':>5}  {'PD0_Shifts':>14}  {'All_Shifts':>14}  {'IMU':>8}")
    print(f"  {'-'*(W-2)}")
    for i, s in enumerate(summaries, 1):
        sid  = f"S{i:02d}"
        pd0  = "Yes" if s["PD0"] else "No"
        pd1  = "Yes" if s["PD1"] else "No"
        pd2  = "Yes" if s["PD2"] else "No"
        pd3  = "Yes" if s["PD3"] else "No"
        hr   = f"{s['ppg_hr']:.1f}" if not np.isnan(s["ppg_hr"]) else "--"

        n0 = s["pd0_shifts"]
        pd0_shift_str = f"Warning ({n0})" if n0 > 0 else "0"

        nt = s["total_shifts"]
        shift_str = f"Warning ({nt})" if nt > 0 else "0"

        imu_status  = s.get("imu_status", "OK")
        imu_display = "OK" if imu_status == "OK" else "Warning"

        print(f"  {sid:<5} {hr:>7}  "
              f"{pd0:>5} {pd1:>5} {pd2:>5} {pd3:>5}  "
              f"{s['n_pass']:>2}/4   {pd0_shift_str:>14}  {shift_str:>14}  {imu_display:>8}")
    print(f"{'='*W}\n")


# ---------------- Main ----------------
def main():
    # Find all aligned files across subject folders
    aligned_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in sorted(files):
            if f.endswith("_fnirs_aligned.csv"):
                subject = f.replace("_fnirs_aligned.csv", "")
                aligned_files.append((subject, os.path.join(root, f)))

    if not aligned_files:
        print(f"No *_fnirs_aligned.csv files found under: {DATA_DIR}")
        return

    print(f"\nFound {len(aligned_files)} subject(s) to process")

    summaries = []
    for participant, data_path in sorted(aligned_files):
        try:
            summary = process_subject(data_path, participant)
            summaries.append(summary)
        except Exception as e:
            print(f"  ERROR processing {participant}: {e}")

    if summaries:
        print_cross_subject_summary(summaries)


if __name__ == "__main__":
    main()