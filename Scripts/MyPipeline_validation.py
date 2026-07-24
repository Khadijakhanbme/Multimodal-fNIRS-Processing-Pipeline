"""
Layer 1 Pipeline Validation — Python vs Homer3 on Artinis SNIRF
================================================================
Applies the same preprocessing logic as the custom sensor pipeline
(OD conversion, bandpass filtering, MBLL, block averaging) to Artinis
SNIRF data, then compares output against Homer3 block-averaged HRFs.

No motion correction in this pass (clean subjects only).
No SSR (Artinis has no short-separation channels).
No PPG (Artinis has no PPG).

Steps:
  1. Load Artinis SNIRF (h5py)
  2. Intensity → OD (log ratio, baseline-referenced)
  3. Bandpass filter (0.01–0.5 Hz, matching Homer3)
  4. MBLL → HbO / HbR (DPF = 6.0 for both λ, matching Homer3/Artinis)
  5. Block average per condition (baseline corrected, -5 to 0 s)
  6. Cross-subject comparison plots

Usage:
  Update SUBJECTS paths below, then run.
"""

import os
import warnings
import numpy as np
import h5py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
import scipy.io as sio

# ============================================================
# CONFIG — SUBJECTS
# ============================================================
_BASE   = r"D:\EMMBIOME- Course work\4th Semester\Thesis- VUB\Internship- VUB\fNIRS_PIPELINE\rawdata\2nd Trial\artinis data"
_HOMER3 = r"D:\EMMBIOME- Course work\4th Semester\Thesis- VUB\Internship- VUB\fNIRS_PIPELINE\rawdata\2nd Trial\Artinis data\derivatives\homer"

SUBJECTS = [
    {
        "name":   "Malin",
        "snirf":  os.path.join(_BASE,   "Malin_artinis",    "malin_aligned.snirf"),
        "homer3": os.path.join(_HOMER3, "Malin_artinis",    "malin_aligned_export.mat"),
        "dpf":    6.0,   # matches Homer3 ppf for this subject
    },
    {
        "name":   "Jonathan",
        "snirf":  os.path.join(_BASE,   "Jonathan_artinis", "jonathan_aligned.snirf"),
        "homer3": os.path.join(_HOMER3, "Jonathan_artinis", "jonathan_aligned_export.mat"),
        "dpf":    5.9,   # matches Homer3 ppf for this subject
    },
    {
        "name":   "Maryum",
        "snirf":  os.path.join(_BASE,   "Maryum_artinis",   "maryum_aligned.snirf"),
        "homer3": os.path.join(_HOMER3, "Maryum_artinis",   "maryum_aligned_export.mat"),
        "dpf":    6.0,   # matches Homer3 ppf for this subject
    },
    {
        "name":   "Taimoor",
        "snirf":  os.path.join(_BASE,   "Taimoor_artinis",  "Taimoor_aligned.snirf"),
        "homer3": os.path.join(_HOMER3, "Taimoor_artinis",  "Taimoor_aligned_export.mat"),
        "dpf":    5.9,   # matches Homer3 ppf for this subject
    },
    # Add 5th subject here if available:
    # {"name": "Name", "snirf": ..., "homer3": ...},
]

# Homer3 condition order in dcAvg dim 3 — must match stim order Homer3 processed.
# Default: Homer3 sorts stim alphabetically → "0-back"=0, "2-back"=1.
# If your results look flipped, swap these.
HOMER3_COND_MAP = {0: "0-back", 1: "2-back"}
HOMER3_HB_MAP   = {0: "HbO",    1: "HbR"}

# Per-subject colors for overlay plots
SUBJ_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

# Artinis wavelengths (from SNIRF probe)
WL1 = 764   # nm
WL2 = 845   # nm

# DPF — from Artinis OxySoft age-based formula (same as Homer3 ppf setting)
DPF_WL1 = 6.0
DPF_WL2 = 6.0

# Extinction coefficients — Prahl (omlc.org/spectra)
# Units: mM^-1 cm^-1, base-10 log convention → converted to natural-log
_PRAHL_LOG10 = {
    764: (0.4677, 1.3491),   # interpolated for 764 nm
    845: (1.0370, 0.6998),   # interpolated for 845 nm
}
EPS = {
    wl: (eo * np.log(10), er * np.log(10))
    for wl, (eo, er) in _PRAHL_LOG10.items()
}

# Filter settings (matching Homer3)
BP_LO = 0.01   # Hz
BP_HI = 0.5    # Hz

# Epoch settings (matching Homer3 block average)
PRE_S  = 5.0    # seconds before task onset
POST_S = 35.0   # seconds after task onset
BASELINE_FROM = -5.0   # baseline correction window start
BASELINE_TO   =  0.0   # baseline correction window end

# Channel labels for Artinis Brite Lite
# Based on probe: Src-Det pairs, ML indices in SNIRF
CH_LABELS = [
    "Rx1-Tx1",  # Ch1: Src1-Det1
    "Rx1-Tx2",  # Ch2: Src2-Det1
    "Rx2-Tx1",  # Ch3: Src1-Det2
    "Rx2-Tx2",  # Ch4: Src2-Det2
    "Rx3-Tx3",  # Ch5: Src3-Det3
    "Rx3-Tx4",  # Ch6: Src4-Det3
    "Rx4-Tx3",  # Ch7: Src3-Det4
    "Rx4-Tx4",  # Ch8: Src4-Det4
]
CH_SIDES = ["Right", "Right", "Right", "Right", "Left", "Left", "Left", "Left"]


# ============================================================
# STEP 1 — Load SNIRF
# ============================================================
def load_snirf(path):
    """Load Artinis SNIRF file. Returns time, intensity data, channel map, and stim events."""
    with h5py.File(path, "r") as f:
        time = f["nirs/data1/time"][()].flatten()
        data = f["nirs/data1/dataTimeSeries"][()]
        wls  = f["nirs/probe/wavelengths"][()].flatten()

        # Build channel map: list of (src, det, wl_idx, wl_nm, col_idx)
        ml_keys = sorted(
            [k for k in f["nirs/data1"].keys() if k.startswith("measurementList")],
            key=lambda x: int(x.replace("measurementList", ""))
        )
        channels = []
        for mk in ml_keys:
            col_idx = int(mk.replace("measurementList", "")) - 1
            src = int(f[f"nirs/data1/{mk}/sourceIndex"][()])
            det = int(f[f"nirs/data1/{mk}/detectorIndex"][()])
            wl_idx = int(f[f"nirs/data1/{mk}/wavelengthIndex"][()])
            channels.append({
                "src": src, "det": det,
                "wl_idx": wl_idx, "wl_nm": int(wls[wl_idx - 1]),
                "col": col_idx
            })

        # Load stimulus events
        stim_events = {}
        stim_keys = sorted([k for k in f["nirs"].keys() if k.startswith("stim")])
        for sk in stim_keys:
            name = f[f"nirs/{sk}/name"][()].decode()
            sdata = f[f"nirs/{sk}/data"][()]
            stim_events[name] = sdata  # (N_events, 3): onset, duration, amplitude

    fs = 1.0 / np.median(np.diff(time))
    print(f"[load]   {len(time)} samples, Fs = {fs:.2f} Hz, duration = {time[-1]:.1f} s")
    print(f"[load]   {len(channels)} measurement lists, wavelengths = {wls}")
    print(f"[load]   Stimulus groups: {list(stim_events.keys())}")

    return time, data, channels, stim_events, fs


def group_channels_by_pair(channels):
    """Group measurement lists into source-detector pairs with two wavelengths each.
    Returns list of dicts: {src, det, wl1_col, wl2_col, label, side}
    """
    pairs = {}
    for ch in channels:
        key = (ch["src"], ch["det"])
        if key not in pairs:
            pairs[key] = {}
        pairs[key][ch["wl_nm"]] = ch["col"]

    ch_list = []
    ch_idx = 0
    for (src, det), wl_map in sorted(pairs.items()):
        if WL1 in wl_map and WL2 in wl_map:
            ch_list.append({
                "src": src, "det": det,
                "wl1_col": wl_map[WL1],
                "wl2_col": wl_map[WL2],
                "label": CH_LABELS[ch_idx] if ch_idx < len(CH_LABELS) else f"S{src}-D{det}",
                "side": CH_SIDES[ch_idx] if ch_idx < len(CH_SIDES) else "Unknown",
            })
            ch_idx += 1

    print(f"[pairs]  {len(ch_list)} source-detector pairs with both wavelengths")
    return ch_list


# ============================================================
# STEP 2 — Intensity to OD (same logic as custom pipeline Step 3.2)
# ============================================================
def intensity_to_od(intensity, baseline_samples=None):
    """
    Convert intensity to optical density: OD = -log(I / I0)
    I0 = mean intensity over baseline period (or full recording if not specified).
    Same approach as custom pipeline: uses median of baseline.
    """
    if baseline_samples is not None:
        I0 = np.median(intensity[baseline_samples])
    else:
        I0 = np.mean(intensity)

    I0 = max(I0, 1e-12)
    od = -np.log(np.clip(intensity, 1e-12, None) / I0)
    return od


# ============================================================
# STEP 3 — Bandpass filter (same as custom pipeline)
# ============================================================
def bandpass(sig, lo, hi, fs, order=3):
    """Butterworth bandpass filter — identical to custom pipeline."""
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


# ============================================================
# STEP 4 — MBLL (same logic as custom pipeline MBLL.py)
# ============================================================
def mbll_two_wavelength(od_wl1, od_wl2, wl1, wl2, sds_cm, dpf1, dpf2):
    """
    Solve MBLL for HbO and HbR using two wavelengths.
    Same implementation as custom pipeline.

        ΔOD(λ) = (ε_HbO(λ)·ΔC_HbO + ε_HbR(λ)·ΔC_HbR) · SDS · DPF(λ)

    Returns ΔC_HbO, ΔC_HbR in micromolar (µM).
    """
    eHbO_1, eHbR_1 = EPS[wl1]
    eHbO_2, eHbR_2 = EPS[wl2]

    L1 = sds_cm * dpf1
    L2 = sds_cm * dpf2

    A = np.array([
        [eHbO_1 * L1, eHbR_1 * L1],
        [eHbO_2 * L2, eHbR_2 * L2]
    ])

    OD = np.vstack([od_wl1, od_wl2])   # (2, N)
    C  = np.linalg.inv(A) @ OD          # (2, N), units: mM
    HbO_uM = C[0] * 1000.0              # convert mM → µM
    HbR_uM = C[1] * 1000.0
    return HbO_uM, HbR_uM


# ============================================================
# STEP 5 — Block Average (same logic as custom pipeline blockAverage.py)
# ============================================================
def extract_epochs(t, y, onsets, pre_s, post_s):
    """Extract fixed-length windows around each onset. Same as custom pipeline."""
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    n_pre  = int(round(pre_s * fs))
    n_post = int(round(post_s * fs))
    epoch_time = np.arange(-n_pre, n_post) * dt

    epochs = []
    skipped = 0
    for onset in onsets:
        i_on = np.searchsorted(t, onset)
        i0 = i_on - n_pre
        i1 = i_on + n_post
        if i0 < 0 or i1 > len(y):
            skipped += 1
            continue
        ep = y[i0:i1]
        if len(ep) == len(epoch_time):
            epochs.append(ep)
        else:
            skipped += 1

    return np.asarray(epochs), epoch_time, skipped


def baseline_correct(epochs, epoch_time, bl_from, bl_to):
    """Block-wise baseline correction. Same as custom pipeline."""
    if epochs.size == 0:
        return epochs
    bl_mask = (epoch_time >= bl_from) & (epoch_time < bl_to)
    bl = np.nanmean(epochs[:, bl_mask], axis=1, keepdims=True)
    return epochs - bl


def block_average(t, y, onsets, pre_s, post_s, bl_from, bl_to):
    """Full block average pipeline. Returns (mean, sem, epoch_time, n_epochs)."""
    epochs, ep_time, skipped = extract_epochs(t, y, onsets, pre_s, post_s)
    if epochs.size == 0:
        return None, None, None, 0
    epochs = baseline_correct(epochs, ep_time, bl_from, bl_to)
    mean = np.nanmean(epochs, axis=0)
    sem  = np.nanstd(epochs, axis=0, ddof=1) / np.sqrt(epochs.shape[0]) if epochs.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem, ep_time, epochs.shape[0]


# ============================================================
# Homer3 comparison
# ============================================================
def load_homer3_export(mat_path):
    """
    Load Homer3 block-average export produced by the MATLAB export script.
    Expects variables: dcAvg [nT×2×nCh×nCond], tHRF [nT], nTrials [nCond].
    Returns dict or None if file missing / unreadable.
    """
    if not os.path.exists(mat_path):
        return None
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        dcAvg   = mat["dcAvg"]    # (nT, 2, nCh, nCond)
        tHRF    = np.atleast_1d(mat["tHRF"]).flatten()
        nTrials = np.atleast_1d(mat["nTrials"]).flatten()

        # Ensure 4-D: some squeeze_me may collapse dims
        if dcAvg.ndim == 3:       # single condition
            dcAvg = dcAvg[:, :, :, np.newaxis]
        if dcAvg.ndim == 2:       # single condition + single channel
            dcAvg = dcAvg[:, :, np.newaxis, np.newaxis]

        return {"tHRF": tHRF, "dcAvg": dcAvg, "nTrials": nTrials}
    except Exception as e:
        print(f"  [homer3] could not load {os.path.basename(mat_path)}: {e}")
        return None


def compare_with_homer3(py_results, homer3, subj_info):
    """
    Quantitative comparison: Python pipeline vs Homer3 block averages.
    Computes per-channel Pearson r, RMSE, peak-amplitude ratio for HbO.
    Returns metrics list and produces two figures.
    """
    if homer3 is None:
        return []

    tH    = homer3["tHRF"]          # Homer3 time axis
    dcAvg = homer3["dcAvg"]         # (nT, 2, nCh, nCond)
    nCh_h3 = dcAvg.shape[2]
    nCond  = dcAvg.shape[3]

    ch_pairs = py_results["ch_pairs"]
    name     = py_results["name"]
    metrics  = []

    # ── Figure A: per-channel overlay (right channels, HbO) ─────────────────
    right_idx = list(range(0, min(4, len(ch_pairs))))
    cond_keys = [HOMER3_COND_MAP[c] for c in range(nCond) if c in HOMER3_COND_MAP]

    fig_a, axes_a = plt.subplots(
        len(cond_keys), len(right_idx),
        figsize=(4.5 * len(right_idx), 3.5 * len(cond_keys)),
        sharex=True, sharey="row", squeeze=False,
    )

    for row, cond_name in enumerate(cond_keys):
        h3_cond_idx = next(k for k, v in HOMER3_COND_MAP.items() if v == cond_name)
        for col, ch_i in enumerate(right_idx):
            ax = axes_a[row][col]

            # Python curve
            entry = py_results["ba"].get((ch_i, cond_name, "HbO"))
            if entry and entry[0] is not None:
                py_mean, py_sem, ep_time, _ = entry
                ax.fill_between(ep_time, py_mean - py_sem, py_mean + py_sem,
                                color="#e6194b", alpha=0.15)
                ax.plot(ep_time, py_mean, color="#e6194b", lw=2.0, label="Python")
            else:
                ep_time = tH
                py_mean = None

            # Homer3 curve (interpolate onto Python time grid if needed)
            if ch_i < nCh_h3:
                h3_curve = dcAvg[:, 0, ch_i, h3_cond_idx] * 1e6  # M → µM
                grids_match = (py_mean is not None
                               and len(tH) == len(ep_time)
                               and np.allclose(tH, ep_time, atol=0.01))
                if py_mean is not None and not grids_match:
                    h3_curve = np.interp(ep_time, tH, h3_curve)
                    t_plot = ep_time
                else:
                    t_plot = tH
                ax.plot(t_plot, h3_curve, color="#4363d8", lw=2.0,
                        ls="--", label="Homer3")

                # Metrics
                if py_mean is not None:
                    common = np.interp(ep_time, tH, dcAvg[:, 0, ch_i, h3_cond_idx] * 1e6)
                    mask   = np.isfinite(py_mean) & np.isfinite(common)
                    if mask.sum() > 5:
                        r, _   = pearsonr(py_mean[mask], common[mask])
                        rmse   = float(np.sqrt(np.mean((py_mean[mask] - common[mask])**2)))
                        pk_py  = float(np.max(py_mean))
                        pk_h3  = float(np.max(common))
                        metrics.append({
                            "Subject":  name,
                            "Ch":       f"Ch{ch_i+1}",
                            "Label":    ch_pairs[ch_i]["label"] if ch_i < len(ch_pairs) else f"Ch{ch_i+1}",
                            "Cond":     cond_name,
                            "Hb":       "HbO",
                            "r":        r,
                            "RMSE_µM":  rmse,
                            "Peak_Py":  pk_py,
                            "Peak_H3":  pk_h3,
                            "Peak_ratio": pk_py / pk_h3 if pk_h3 != 0 else np.nan,
                        })

            ax.axvspan(0, 27.5, color="orange", alpha=0.08, lw=0)
            ax.axvline(0,  color="k", lw=0.5, ls="--", alpha=0.5)
            ax.axhline(0,  color="k", lw=0.3)
            ax.set_xlim(-PRE_S, POST_S)
            ax.set_title(f"Ch{ch_i+1} ({ch_pairs[ch_i]['label'] if ch_i < len(ch_pairs) else ''})",
                         fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)
            if row == len(cond_keys) - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{cond_name}\nΔHbO (µM)", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, framealpha=0.85)

    fig_a.suptitle(
        f"{name} — Python vs Homer3 HbO overlay (right channels)\n"
        "Red = Python pipeline   Blue dashed = Homer3",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ── Figure B: scatter — Python peak vs Homer3 peak, all channels ─────────
    if metrics:
        df_m = {k: [d[k] for d in metrics] for k in metrics[0]}
        fig_b, axes_b = plt.subplots(1, len(cond_keys),
                                     figsize=(5 * len(cond_keys), 5),
                                     squeeze=False)
        for col, cond_name in enumerate(cond_keys):
            ax = axes_b[0][col]
            rows = [d for d in metrics if d["Cond"] == cond_name]
            if not rows:
                continue
            px = [d["Peak_Py"] for d in rows]
            ph = [d["Peak_H3"] for d in rows]
            rs = [d["r"]       for d in rows]
            lbls = [d["Label"] for d in rows]

            sc = ax.scatter(ph, px, c=rs, cmap="RdYlGn", vmin=-1, vmax=1,
                            s=90, zorder=3, edgecolors="k", linewidths=0.5)
            for xi, yi, lbl in zip(ph, px, lbls):
                ax.annotate(lbl, (xi, yi), textcoords="offset points",
                            xytext=(5, 4), fontsize=8)

            lim = max(abs(np.array(px + ph))) * 1.15
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5,
                    label="y = x (perfect)")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Homer3 peak HbO (µM)", fontsize=10)
            ax.set_ylabel("Python peak HbO (µM)", fontsize=10)
            ax.set_title(f"{cond_name}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.colorbar(sc, ax=ax, label="Pearson r")

        fig_b.suptitle(
            f"{name} — Python vs Homer3: peak amplitude scatter\n"
            "Points on diagonal = perfect agreement   Colour = correlation",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])

    return metrics


# ============================================================
# Per-subject processing
# ============================================================
def process_subject(subj_info):
    """Run full pipeline for one subject. Returns results dict."""
    name = subj_info["name"]
    path = subj_info["snirf"]

    print(f"\n{'─'*60}")
    print(f"  Subject: {name}")
    print(f"{'─'*60}")

    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return None

    time, data, channels, stim_events, fs = load_snirf(path)
    ch_pairs = group_channels_by_pair(channels)

    SDS_CM = 3.0
    dpf = float(subj_info.get("dpf", 6.0))   # per-subject, matches Homer3 ppf

    all_onsets = []
    for ev_name, sdata in stim_events.items():
        if ev_name != "rest":
            all_onsets.extend(sdata[:, 0].tolist())
    first_task    = min(all_onsets) if all_onsets else 30.0
    baseline_end  = max(first_task - 5.0, 10.0)
    baseline_samp = np.where(time < baseline_end)[0]
    print(f"[baseline] t < {baseline_end:.1f} s  ({len(baseline_samp)} samples)")

    # OD → Bandpass → MBLL
    hbo_ts = {}
    hbr_ts = {}
    print(f"{'Ch':>4} | {'Label':>10} | {'Side':>6} | {'σ(HbO)':>8} | {'σ(HbR)':>8}")
    print("-" * 50)
    for i, ch in enumerate(ch_pairs):
        od1 = intensity_to_od(data[:, ch["wl1_col"]], baseline_samp)
        od2 = intensity_to_od(data[:, ch["wl2_col"]], baseline_samp)
        od1_bp = bandpass(od1, BP_LO, BP_HI, fs)
        od2_bp = bandpass(od2, BP_LO, BP_HI, fs)
        hbo, hbr = mbll_two_wavelength(od1_bp, od2_bp, WL1, WL2, SDS_CM, dpf, dpf)
        hbo_ts[i] = hbo
        hbr_ts[i] = hbr
        print(f"{i+1:>4} | {ch['label']:>10} | {ch['side']:>6} | "
              f"{np.std(hbo):>8.3f} | {np.std(hbr):>8.3f}")

    # Block averages
    conditions = {n: d[:, 0] for n, d in stim_events.items()
                  if n in ("0-back", "2-back")}
    print(f"\n[epochs]  {', '.join(f'{n}: {len(v)} blocks' for n, v in conditions.items())}")

    ba = {}  # (ch_idx, cond, hb) -> (mean, sem, ep_time, n)
    for i in range(len(ch_pairs)):
        for cond, onsets in conditions.items():
            for hb, ts in [("HbO", hbo_ts[i]), ("HbR", hbr_ts[i])]:
                ba[(i, cond, hb)] = block_average(
                    time, ts, onsets, PRE_S, POST_S, BASELINE_FROM, BASELINE_TO)

    # Hemisphere averages (Right = Ch0-3, Left = Ch4-7)
    right_idx = list(range(0, min(4, len(ch_pairs))))
    left_idx  = list(range(4, min(8, len(ch_pairs))))

    hem_avg = {}  # (hem, cond, hb) -> (mean, ep_time)
    for hem, idx_list in [("Right", right_idx), ("Left", left_idx)]:
        if not idx_list:
            continue
        for cond in conditions:
            for hb in ("HbO", "HbR"):
                curves = [ba[(i, cond, hb)][0] for i in idx_list
                          if ba[(i, cond, hb)][0] is not None]
                ep_time = next((ba[(i, cond, hb)][2] for i in idx_list
                                if ba[(i, cond, hb)][2] is not None), None)
                if curves and ep_time is not None:
                    hem_avg[(hem, cond, hb)] = (np.mean(curves, axis=0), ep_time)

    return {
        "name":     name,
        "ch_pairs": ch_pairs,
        "ba":       ba,
        "hem_avg":  hem_avg,
        "conditions": list(conditions.keys()),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  Layer 1 Pipeline Validation — Multi-Subject (Artinis)")
    print("=" * 70)

    all_subj     = []
    all_metrics  = []   # Homer3 comparison metrics across subjects

    for s in SUBJECTS:
        res = process_subject(s)
        if res is None:
            continue
        all_subj.append(res)

        # Homer3 comparison (only if export file exists)
        h3 = load_homer3_export(s.get("homer3", ""))
        if h3 is not None:
            print(f"\n  [Homer3] Comparing {s['name']} ...")
            m = compare_with_homer3(res, h3, s)
            all_metrics.extend(m)
        else:
            print(f"  [Homer3] No export file for {s['name']} — skipping comparison"
                  f"\n           (run the MATLAB export script first)")

    if not all_subj:
        print("No subjects processed — check file paths.")
        return

    conds = ["0-back", "2-back"]
    hems  = ["Right", "Left"]

    # ── Figure 1: Right channels individually — all subjects overlaid ────────
    RIGHT_IDX = list(range(0, 4))  # Ch1–Ch4 = right hemisphere

    # Get channel labels from first subject that has all 4 right channels
    _ref_pairs = next((s["ch_pairs"] for s in all_subj
                       if len(s["ch_pairs"]) >= 4), all_subj[0]["ch_pairs"])
    right_labels = [_ref_pairs[i]["label"] if i < len(_ref_pairs)
                    else f"Ch{i+1}" for i in RIGHT_IDX]

    fig1, axes1 = plt.subplots(
        len(conds), len(RIGHT_IDX),
        figsize=(4.5 * len(RIGHT_IDX), 4 * len(conds)),
        sharex=True, sharey="row",
        squeeze=False,
    )

    for row, cond in enumerate(conds):
        for col, ch_i in enumerate(RIGHT_IDX):
            ax = axes1[row][col]
            for si, subj in enumerate(all_subj):
                entry = subj["ba"].get((ch_i, cond, "HbO"))
                if entry is None or entry[0] is None:
                    continue
                mean, sem, ep_time, n = entry
                color = SUBJ_COLORS[si % len(SUBJ_COLORS)]
                ax.fill_between(ep_time, mean - sem, mean + sem,
                                color=color, alpha=0.12)
                ax.plot(ep_time, mean, color=color, lw=2.0,
                        label=subj["name"])
                # HbR faint dashed
                entry_r = subj["ba"].get((ch_i, cond, "HbR"))
                if entry_r and entry_r[0] is not None:
                    ax.plot(ep_time, entry_r[0], color=color,
                            lw=1.0, ls="--", alpha=0.4)

            ax.axvspan(0, 27.5, color="orange", alpha=0.08, lw=0)
            ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax.axhline(0, color="k", lw=0.3)
            ax.set_xlim(-PRE_S, POST_S)
            ax.set_title(f"Ch{ch_i+1}: {right_labels[col]}",
                         fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)
            if row == len(conds) - 1:
                ax.set_xlabel("Time relative to onset (s)", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{cond}\nΔ[Hb] (µM)", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, framealpha=0.85, loc="upper right")

    fig1.suptitle(
        "Right hemisphere — individual channels — all subjects\n"
        "HbO (solid) · HbR (dashed, faint) · shaded = ±1 SEM · orange = task",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ── Figure 2: Peak HbO per subject — grouped bar chart ───────────────────
    n_subj = len(all_subj)
    fig2, axes2 = plt.subplots(1, len(hems), figsize=(6 * len(hems), 5), sharey=True)
    if len(hems) == 1:
        axes2 = [axes2]

    bar_w = 0.35
    x     = np.arange(n_subj)

    for col, hem in enumerate(hems):
        ax = axes2[col]
        peaks_0 = []
        peaks_2 = []
        labels  = []
        for subj in all_subj:
            for cond, store in [("0-back", peaks_0), ("2-back", peaks_2)]:
                key = (hem, cond, "HbO")
                if key in subj["hem_avg"]:
                    mean, _ = subj["hem_avg"][key]
                    store.append(float(np.max(mean)))
                else:
                    store.append(np.nan)
            labels.append(subj["name"])

        ax.bar(x - bar_w / 2, peaks_0, bar_w,
               label="0-back", color="#2196F3", alpha=0.85, edgecolor="white")
        ax.bar(x + bar_w / 2, peaks_2, bar_w,
               label="2-back", color="#F44336", alpha=0.85, edgecolor="white")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
        ax.set_title(f"{hem} hemisphere", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        if col == 0:
            ax.set_ylabel("Peak HbO (µM)", fontsize=10)

    fig2.suptitle(
        "Peak HbO per subject per condition — hemisphere average\n"
        "Pipeline validation: expect 2-back ≥ 0-back (working memory load effect)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    # ── Terminal summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Summary — Peak HbO (µM), hemisphere-averaged")
    print(f"{'='*70}")
    print(f"  {'Subject':<12}  {'Right 0b':>9}  {'Right 2b':>9}  {'Left 0b':>9}  {'Left 2b':>9}  {'2b>0b?':>7}")
    print(f"  {'─'*60}")
    for subj in all_subj:
        def pk(hem, cond):
            key = (hem, cond, "HbO")
            if key not in subj["hem_avg"]:
                return np.nan
            return float(np.max(subj["hem_avg"][key][0]))
        r0, r2 = pk("Right", "0-back"), pk("Right", "2-back")
        l0, l2 = pk("Left",  "0-back"), pk("Left",  "2-back")
        ok = "Yes" if (r2 > r0 or l2 > l0) else "No"
        print(f"  {subj['name']:<12}  {r0:>9.4f}  {r2:>9.4f}  {l0:>9.4f}  {l2:>9.4f}  {ok:>7}")
    print(f"{'='*70}\n")

    # ── Homer3 quantitative summary ───────────────────────────────────────────
    if all_metrics:
        W = 88
        print(f"\n{'='*W}")
        print("  Python vs Homer3 — Quantitative Comparison (HbO, right channels)")
        print(f"{'='*W}")
        print(f"  {'Subject':<10} {'Ch':>5} {'Chan':>10} {'Cond':>8}  "
              f"{'r':>6}  {'RMSE µM':>8}  {'Pk Py':>8}  {'Pk H3':>8}  {'Ratio':>6}")
        print(f"  {'─'*(W-2)}")
        for m in all_metrics:
            print(f"  {m['Subject']:<10} {m['Ch']:>5} {m['Label']:>10} {m['Cond']:>8}  "
                  f"{m['r']:>6.3f}  {m['RMSE_µM']:>8.4f}  "
                  f"{m['Peak_Py']:>8.4f}  {m['Peak_H3']:>8.4f}  {m['Peak_ratio']:>6.3f}")
        # Cross-subject average
        rs   = [m["r"]          for m in all_metrics]
        rms  = [m["RMSE_µM"]   for m in all_metrics]
        rats = [m["Peak_ratio"] for m in all_metrics if np.isfinite(m["Peak_ratio"])]
        print(f"  {'─'*(W-2)}")
        print(f"  {'MEAN':>10}  {'':>5} {'':>10} {'':>8}  "
              f"{np.mean(rs):>6.3f}  {np.mean(rms):>8.4f}  "
              f"{'':>8}  {'':>8}  {np.mean(rats):>6.3f}")
        print(f"{'='*W}")
        print("  r = Pearson correlation of HRF time course")
        print("  RMSE = root-mean-square error (µM)")
        print("  Ratio = Python peak / Homer3 peak  (1.0 = perfect match)")
        print(f"{'='*W}\n")
    else:
        print("\n  No Homer3 export files found — run the MATLAB export script to enable comparison.\n")


if __name__ == "__main__":
    main()