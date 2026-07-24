"""
10_HRF_grid_subject_by_channel.py
=================================
Fig-4-style grid of CORRECTED block-averaged hemodynamic responses.

  rows    = participants (anonymized S01, S02, ... ; SSR subjects first, then PPG-only)
  columns = channels  PD1 (25 mm) | PD2 (30 mm) | PD3 (40 mm)
  panel   = corrected ΔHbO (red) + ΔHbR (blue), mean ± SEM across task blocks

"Corrected" = the subject's best-available systemic correction, applied at the GLM
level and projected back out of the signal (corrected = residual + task component):
  - Strand 1 (PD0 survived): short-separation regression (PD0 OD @740+850 nm)
  - Strand 2 (PD0 dropped):   PPG nuisance regression  (fallback)

PD3 is included even though it survives QC in ~one participant — the single
populated panel (with the rest "not retained") documents the 40 mm channel yield.

Set INCLUDE_PPG_ONLY = False for an SSR-only version (cleaner, for the main text);
keep True for the complete appendix version. Display only — no figures saved.

This is the clean morphology figure. It deliberately omits the raw trace and the
GLM-fit overlay (those live in the per-subject raw-vs-corrected figure from the
GLM script); here we show only the corrected responses for comparability.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammaln
from scipy.signal import butter, filtfilt

# ── Config ───────────────────────────────────────────────────────────────────────
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_SCRIPT_DIR, "..", "rawdata", "2nd Trial", "Multimodal_data")

FS             = 20.0
DRIFT_CUTOFF_S = 200.0
WL1, WL2       = 740, 850
SS_PD          = "PD0"
PPG_COL        = "PPG (635nm)"
PPG_BP_LO, PPG_BP_HI = 0.01, 0.20
HB_TYPES       = ["HbO", "HbR"]
TARGET_PDS     = ["PD1", "PD2", "PD3"]          # PD3 kept to show single-subject yield
PD_LABELS      = {"PD1": "PD1 (25 mm)", "PD2": "PD2 (30 mm)", "PD3": "PD3 (40 mm)"}

INCLUDE_PPG_ONLY = True    # False → SSR-corrected subjects only (main-text version)

# Block-average epoching
PRE_S, POST_S              = 5.0, 30.0
BASELINE_FROM, BASELINE_TO = -5.0, 0.0
TASK_DUR_S                 = 27.0

SDS = {"PD0": 0.75, "PD1": 2.50, "PD2": 3.00, "PD3": 3.50}

SUBJECT_AGES = {
    "youssef": 20, "esin": 23, "lina": 20, "maram": 22, "yasmine": 23,
    "elmehdi": 22, "ali": 25, "sami": 22, "nouha": 23, "sheharyar": 26,
    "aqsa": 27, "irem": 25, "issam": 22,
}

_PRAHL = {740: (0.4464, 1.3837), 850: (1.0507, 0.6912)}
EPS    = {wl: (eo * np.log(10), er * np.log(10)) for wl, (eo, er) in _PRAHL.items()}


# ── Helpers ──────────────────────────────────────────────────────────────────────
def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def bandpass(sig, fs, lo, hi, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.asarray(sig, dtype=float))

def scholkmann_dpf(wl_nm, age):
    return (223.3 + 0.05624 * (age ** 0.8493) - 5.723e-7 * (wl_nm ** 3)
            + 0.001245 * (wl_nm ** 2) - 0.9025 * wl_nm)

def _od(wl, pd_name):  return f"OD ({wl}nm) {pd_name}"
def _odp(wl, pd_name): return f"ODp ({wl}nm) {pd_name}"

def _long_pds(df):
    pds = set()
    for c in df.columns:
        if c.startswith("OD (") and "PD0" not in c and "PD" in c:
            try:
                pds.add("PD" + c.split("PD")[1])
            except Exception:
                pass
    return sorted(p for p in pds if p in TARGET_PDS)

def long_od_frame(df):
    out = {}
    for pd_name in _long_pds(df):
        for wl in (WL1, WL2):
            lc = _od(wl, pd_name)
            if lc in df.columns:
                out[_odp(wl, pd_name)] = df[lc].to_numpy(float)
    return pd.DataFrame(out)

def run_mbll(df_odp, age):
    pds   = sorted({"PD" + c.split("PD")[1] for c in df_odp.columns
                    if c.startswith("ODp (") and "PD" in c})
    df_hb = pd.DataFrame()
    eHbO1, eHbR1 = EPS[WL1]; eHbO2, eHbR2 = EPS[WL2]
    for pd_name in pds:
        c1, c2 = _odp(WL1, pd_name), _odp(WL2, pd_name)
        if c1 not in df_odp.columns or c2 not in df_odp.columns:
            continue
        sds = SDS.get(pd_name, 2.5)
        L1, L2 = sds * scholkmann_dpf(WL1, age), sds * scholkmann_dpf(WL2, age)
        A   = np.array([[eHbO1 * L1, eHbR1 * L1], [eHbO2 * L2, eHbR2 * L2]])
        OD  = np.vstack([df_odp[c1].to_numpy(float), df_odp[c2].to_numpy(float)])
        C   = np.linalg.inv(A) @ OD
        df_hb[f"HbO_{pd_name}"] = C[0] * 1000.0
        df_hb[f"HbR_{pd_name}"] = C[1] * 1000.0
    return df_hb


# ── HRF / design / GLM ──────────────────────────────────────────────────────────
def spm_hrf(fs, length_s=32.0):
    dt = 1.0 / fs; t = np.arange(0, length_s, dt)
    def gpdf(t, peak, disp):
        a = peak / disp
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = (a - 1) * np.log(t / disp) - (t / disp) - gammaln(a) - np.log(disp)
        p = np.exp(logp); p[~np.isfinite(p)] = 0.0
        return p
    h = gpdf(t, 6.0, 1.0) - gpdf(t, 16.0, 1.0) / 6.0
    return h / (np.max(np.abs(h)) + 1e-12)

def boxcar_regressor(t, onsets, durations, hrf):
    t = np.asarray(t, dtype=float); dt = np.median(np.diff(t))
    box = np.zeros(len(t))
    for onset, dur in zip(onsets, durations):
        i0 = max(0, min(np.searchsorted(t, onset), len(t)))
        i1 = max(0, min(np.searchsorted(t, onset + dur), len(t)))
        if i1 > i0:
            box[i0:i1] = 1.0
    reg = np.convolve(box, hrf)[:len(t)] * dt
    return reg / (np.max(np.abs(reg)) + 1e-12)

def cosine_drift_basis(t, cutoff_s):
    N = len(t); T = t[-1] - t[0]; K = int(np.floor(2 * T / cutoff_s))
    if K < 1:
        return np.empty((N, 0))
    n = np.arange(N); basis = np.zeros((N, K))
    for k in range(1, K + 1):
        basis[:, k - 1] = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return basis

def build_design(t, ev_2, ev_0, fs, extra=None):
    hrf = spm_hrf(fs); drift = cosine_drift_basis(t, DRIFT_CUTOFF_S); const = np.ones((len(t), 1))
    ev_all = pd.concat([ev_2, ev_0], ignore_index=True).sort_values("onset")
    reg_task = boxcar_regressor(t, ev_all["onset"].to_numpy(), ev_all["duration"].to_numpy(), hrf)
    regs = [reg_task]
    for _, arr in (extra or []):
        regs.append(np.asarray(arr, dtype=float))
    cols = regs + ([drift] if drift.shape[1] > 0 else []) + [const]
    return np.column_stack(cols)   # task is column 0

def fit_betas(X, y):
    """OLS → AR(1) prewhiten → refit. Returns betas (task at index 0)."""
    y = np.asarray(y, dtype=float)
    betas = np.linalg.pinv(X.T @ X) @ X.T @ y
    resid = y - X @ betas
    r = resid - resid.mean(); denom = np.dot(r[:-1], r[:-1])
    rho = float(np.clip(np.dot(r[1:], r[:-1]) / denom, -0.95, 0.95)) if denom > 0 else 0.0
    if abs(rho) > 1e-6:
        Xw, yw = X.copy(), y.copy()
        Xw[1:] -= rho * X[:-1]; yw[1:] -= rho * y[:-1]
        s0 = np.sqrt(1 - rho ** 2); Xw[0] *= s0; yw[0] *= s0
        betas = np.linalg.pinv(Xw.T @ Xw) @ Xw.T @ yw
    return betas


# ── Data loading ──────────────────────────────────────────────────────────────────
def get_events(df, t):
    ev = df["event_name"].astype(str).str.strip().to_numpy()
    idx = np.where(np.concatenate([[True], ev[1:] != ev[:-1]]))[0]
    o2, d2, o0, d0 = [], [], [], []
    for i, start in enumerate(idx):
        label = ev[start]; end = idx[i + 1] - 1 if i + 1 < len(idx) else len(t) - 1
        onset, dur = t[start], t[end] - t[start]
        if   label == "2-back": o2.append(onset); d2.append(dur)
        elif label == "0-back": o0.append(onset); d0.append(dur)
    return (pd.DataFrame({"onset": o2, "duration": d2}),
            pd.DataFrame({"onset": o0, "duration": d0}))

def get_all_onsets(ev_2, ev_0):
    return pd.concat([ev_2, ev_0]).sort_values("onset")["onset"].to_numpy()

def get_ss_od(df, n):
    regs = []
    for wl in (WL1, WL2):
        c = _od(wl, SS_PD)
        if c not in df.columns:
            return None
        x = df[c].to_numpy(float)[:n]
        if np.all(np.isnan(x)) or np.nanstd(x) < 1e-12:
            return None
        regs.append((f"SS{wl}", zscore(x)))
    return regs

def get_ppg(df, n):
    if PPG_COL not in df.columns:
        return None
    ppg = df[PPG_COL].to_numpy(float)[:n]
    if np.all(np.isnan(ppg)) or np.nanstd(ppg) < 1e-12:
        return None
    return zscore(bandpass(zscore(ppg), FS, PPG_BP_LO, PPG_BP_HI))


# ── Epoching ─────────────────────────────────────────────────────────────────────
def extract_epochs(t, y, onsets):
    n_pre, n_post = int(round(PRE_S * FS)), int(round(POST_S * FS))
    dt = np.median(np.diff(t)); ep_t = np.arange(-n_pre, n_post) * dt
    epochs = []
    for onset in onsets:
        i_on = np.searchsorted(t, onset); i0, i1 = i_on - n_pre, i_on + n_post
        if i0 < 0 or i1 > len(y):
            continue
        ep = y[i0:i1]
        if len(ep) == len(ep_t):
            epochs.append(ep)
    return (np.array(epochs) if epochs else np.empty((0, len(ep_t)))), ep_t

def baseline_correct(epochs, ep_t):
    if epochs.size == 0:
        return epochs
    m = (ep_t >= BASELINE_FROM) & (ep_t < BASELINE_TO)
    return epochs - np.nanmean(epochs[:, m], axis=1, keepdims=True)

def mean_sem(epochs):
    mean = np.nanmean(epochs, axis=0)
    sem  = (np.nanstd(epochs, axis=0, ddof=1) / np.sqrt(epochs.shape[0])
            if epochs.shape[0] > 1 else np.zeros_like(mean))
    return mean, sem

def corrected_block_average(df, age):
    """Per channel: corrected (= residual + task component) block average."""
    t = df["time_s"].to_numpy(float); fs = 1.0 / np.median(np.diff(t))
    ev_2, ev_0 = get_events(df, t)
    if ev_2.empty or ev_0.empty:
        return None, None
    onsets = get_all_onsets(ev_2, ev_0)
    hb = run_mbll(long_od_frame(df), age)
    long_pds = sorted([c.replace("HbO_", "") for c in hb.columns if c.startswith("HbO_")])
    if not long_pds:
        return None, None

    ss  = get_ss_od(df, len(t)); ppg = get_ppg(df, len(t))
    has_ss, has_ppg = ss is not None, ppg is not None
    if has_ss:
        extra, corr = ss, "SSR"
    elif has_ppg:
        extra, corr = [("PPG", ppg)], "PPG"
    else:
        extra, corr = [], "none"
    X = build_design(t, ev_2, ev_0, fs, extra); n_min = min(len(t), X.shape[0]); Xf = X[:n_min]

    out = {}
    for pd_name in long_pds:
        pdr = {}
        for hb_t, key in [("HbO", "hbo"), ("HbR", "hbr")]:
            col = f"{hb_t}_{pd_name}"
            if col not in hb.columns:
                continue
            y     = hb[col].to_numpy(float)[:n_min]
            betas = fit_betas(Xf, y)
            y_corr = (y - Xf @ betas) + betas[0] * Xf[:, 0]   # residual + task component
            ep, ep_t = extract_epochs(t[:n_min], y_corr, onsets)
            ep = baseline_correct(ep, ep_t)
            if ep.size == 0:
                continue
            m, s = mean_sem(ep)
            pdr[f"{key}_mean"] = m; pdr[f"{key}_sem"] = s
            pdr["ep_t"] = ep_t; pdr["n_epochs"] = ep.shape[0]
        if pdr:
            out[pd_name] = pdr
    return out, corr


# ── Grid plot ──────────────────────────────────────────────────────────────────────
def plot_grid(ba, sid_order, corr_of):
    pd_cols = [p for p in TARGET_PDS
               if any(p in ba.get(s, {}) for s in sid_order)]
    if not pd_cols:
        print("No channels to plot."); return
    nrows, ncols = len(sid_order), len(pd_cols)

    # Independent y-limits PER CELL: each (subject, PD) autoscales to its own
    # data — PD2/PD3 responses are no longer compressed by PD1 amplitudes.
    cell_ylim = {}
    for sid in sid_order:
        for pd_name in pd_cols:
            pdr = ba.get(sid, {}).get(pd_name)
            if pdr is None:
                cell_ylim[(sid, pd_name)] = (-1, 1)
                continue
            vmax = 0.0
            for key in ("hbo", "hbr"):
                if f"{key}_mean" in pdr:
                    vmax = max(vmax, np.nanmax(np.abs(pdr[f"{key}_mean"])))
            cell_ylim[(sid, pd_name)] = (-1.15 * vmax, 1.15 * vmax) if vmax > 0 else (-1, 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.15 * nrows),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.5, wspace=0.32, left=0.13, right=0.97,
                        top=0.93, bottom=0.05)
    for i, sid in enumerate(sid_order):
        for j, pd_name in enumerate(pd_cols):
            ax = axes[i][j]
            pdr = ba.get(sid, {}).get(pd_name)
            ax.set_xlim(-PRE_S, POST_S)
            ax.set_xticks([0, 15, 30])
            ax.set_ylim(*cell_ylim[(sid, pd_name)])
            if pdr is None:
                ax.text(0.5, 0.5, "not retained", ha="center", va="center",
                        fontsize=6.5, color="#bbbbbb", transform=ax.transAxes)
                ax.set_yticks([])
                ax.tick_params(labelsize=6.5, labelbottom=(i == nrows - 1))
                for sp in ax.spines.values():
                    sp.set_alpha(0.25)
                continue
            ep_t = pdr["ep_t"]
            ax.plot(ep_t, pdr["hbo_mean"], color="#a01010", lw=1.5)
            ax.plot(ep_t, pdr["hbr_mean"], color="#16306b", lw=1.5)
            ax.axvspan(0, TASK_DUR_S, color="orange", alpha=0.07, lw=0)
            ax.axvline(0, color="k", lw=0.4, ls="--", alpha=0.4)
            ax.axhline(0, color="k", lw=0.3)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            # pad=5 pulls tick numbers away from the spine
            ax.tick_params(labelsize=6.5, labelbottom=(i == nrows - 1),
                           labelleft=(j == 0), pad=5)
            if j == 0:
                ax.text(0.05, 0.93, corr_of[sid], transform=ax.transAxes,
                        fontsize=6, color="#888888", va="top")
        axes[i][0].set_ylabel(f"{sid}\nΔ[Hb] (µM)", fontsize=6.5,
                              rotation=90, ha="center", va="center", labelpad=10)
    for j, pd_name in enumerate(pd_cols):
        axes[0][j].set_title(PD_LABELS.get(pd_name, pd_name), fontsize=10, fontweight="bold")
        axes[-1][j].set_xlabel("time (s)", fontsize=8)

    # Save thesis-quality versions before displaying
    out_base = os.path.join(_SCRIPT_DIR, "hrf_grid_subject")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"  Saved: {out_base}.pdf  (vector — use this in thesis)")
    print(f"  Saved: {out_base}.png  (300 DPI raster fallback)")
    plt.show()



# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    folders = sorted([os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d))])
    valid = [f for f in folders
             if os.path.exists(os.path.join(f, f"{os.path.basename(f)}_preprocessed_OD.csv"))
             and SUBJECT_AGES.get(os.path.basename(f).lower()) is not None]
    sid_map = {os.path.basename(f): f"S{i+1:02d}" for i, f in enumerate(valid)}

    print("\nStep 10: Corrected HRF grid (participants × channels: PD1, PD2, PD3)")
    print("  Anonymized IDs: " + ", ".join(sorted(sid_map.values())) + "\n")

    ba, corr_of, strand_of = {}, {}, {}
    for folder in valid:
        name = os.path.basename(folder); sid = sid_map[name]

        age  = SUBJECT_AGES[name.lower()]
        df   = pd.read_csv(os.path.join(folder, f"{name}_preprocessed_OD.csv"), low_memory=False)
        if "time_s" not in df.columns:
            df["time_s"] = np.arange(len(df)) / FS
        try:
            out, corr = corrected_block_average(df, age)
            if out is None:
                print(f"  {sid}: no usable data — skipping"); continue
            ba[sid] = out; corr_of[sid] = corr
            strand_of[sid] = 1 if corr == "SSR" else 2
            print(f"  {sid}: channels {sorted(out.keys())}  [{corr}]")
        except Exception as e:
            import traceback
            print(f"  ERROR — {sid}: {e}"); traceback.print_exc()

    if not ba:
        print("\nNo results."); return

    # Order: SSR subjects first, then PPG-only; optionally drop PPG-only entirely
    s1 = sorted([s for s in ba if strand_of[s] == 1])
    s2 = sorted([s for s in ba if strand_of[s] == 2])
    sid_order = s1 + (s2 if INCLUDE_PPG_ONLY else [])
    if not INCLUDE_PPG_ONLY and s2:
        print(f"\n  (INCLUDE_PPG_ONLY=False → omitting PPG-only subjects: {s2})")
    pd3_subs = [s for s in sid_order if "PD3" in ba[s]]
    print(f"\n  PD3 retained in {len(pd3_subs)} participant(s): {pd3_subs}")
    print(f"  Plotting grid: {len(sid_order)} participants × {len(TARGET_PDS)} channels ...")
    plot_grid(ba, sid_order, corr_of)


if __name__ == "__main__":
    main()