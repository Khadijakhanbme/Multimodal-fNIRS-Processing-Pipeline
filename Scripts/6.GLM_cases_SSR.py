"""
9_GLM_cases_SSR_PPG.py
======================
GLM systemic-correction evaluation for the multimodal fNIRS sensor.

MBLL is run ONLY on the accepted, filtered LONG channels (PD1, PD2) from the
preprocessing stage. PD3 (40 mm) is excluded — it does not survive quality
control. PD0 (short channel) is NEVER converted to haemoglobin; it enters only
as an OD-domain nuisance regressor in the GLM (740 + 850 nm columns), which is
mathematically equivalent to short-separation regression on both chromophores
but keeps PD0 out of the concentration domain.

Two reporting strands (a subject belongs to exactly one, routed automatically by
whether its PD0 survived the upstream coupling-shift check):

  STRAND 1 — PD0 survived → short-separation regression available
    CASE no_ssr    X = [task, drift, const]
    CASE ssr       X = [task, SS740, SS850, drift, const]
    CASE ssr_ppg   X = [task, SS740, SS850, PPG, drift, const]
    Centerpiece: per-subject results + matched-subject group comparison.

  STRAND 2 — PD0 dropped → no short channel
    CASE no_ssr    X = [task, drift, const]
    CASE ppg_only  X = [task, PPG, drift, const]
    Reported SEPARATELY and labelled "best available correction without a short
    channel". Not pooled with Strand 1: different correction method, and these
    are also the worse-contact recordings (two confounds move together).

Reporting choices:
  - t-values are NOT reported (inflated by temporal autocorrelation across many
    samples → misleading). We report beta(task), SE, p-value (fNIRS convention)
    with significance stars, and CNR = |beta(task)| / sigma(residual).
  - Group comparisons use the matched set (subjects present in all of a strand's
    cases) so every case averages the same people; n is annotated per bar.

Plots (displayed only, never saved):
  - Per-subject (first N_PLOT): GLM fit overlay; beta & CNR grouped bars by case
  - Block-averaged HRF + GLM-fit overlay grid (all subjects, PD1 & PD2):
        empirical mean ± SEM (red HbO / blue HbR) with GLM task component
        (black solid HbO, black dashed HbR). Strand-2 (no-SSR) subjects drawn
        dashed and tagged "(no SSR)".
  - Group-mean beta & CNR per channel, by case (Strand 1 matched, and Strand 2)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
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
N_PLOT         = 3
HB_TYPES       = ["HbO", "HbR"]
TARGET_PDS     = ["PD1", "PD2"]      # PD3 (40 mm) excluded by design

# Block-average epoching (matches 9_BlockAverage_GLM_overlay.py)
PRE_S, POST_S       = 5.0, 35.0
BASELINE_FROM, BASELINE_TO = -5.0, 0.0
TASK_DUR_S          = 27.0
NCOLS               = 4              # block-average grid columns

SDS = {"PD0": 0.75, "PD1": 2.50, "PD2": 3.00, "PD3": 3.50}

SUBJECT_AGES = {
    "youssef": 20, "esin": 23, "lina": 20, "maram": 22, "yasmine": 23,
    "elmehdi": 22, "ali": 25, "sami": 22, "nouha": 23, "sheharyar": 26,
    "aqsa": 27, "irem": 25, "issam": 22,
}

_PRAHL = {740: (0.4464, 1.3837), 850: (1.0507, 0.6912)}
EPS    = {wl: (eo * np.log(10), er * np.log(10)) for wl, (eo, er) in _PRAHL.items()}

# Case definitions (single source of truth)
CASE_DEFS = {
    "no_ssr":   {"label": "GLM (no SSR, no PPG)",          "ssr": False, "ppg": False, "color": "#9e9e9e"},
    "ssr":      {"label": "GLM + SSR",                     "ssr": True,  "ppg": False, "color": "#4c78a8"},
    "ssr_ppg":  {"label": "GLM + SSR + PPG",               "ssr": True,  "ppg": True,  "color": "#e45756"},
    "ppg_only": {"label": "GLM + PPG (no short channel)",  "ssr": False, "ppg": True,  "color": "#59a14f"},
}
# Strand 1 ordered none → PPG → SSR → both, so bars read left-to-right as
# increasing correction (makes "SSR carries most of the work" visually obvious).
STRAND1_CASES = ["no_ssr", "ppg_only", "ssr", "ssr_ppg"]
STRAND2_CASES = ["no_ssr", "ppg_only"]


# ── Basic helpers ─────────────────────────────────────────────────────────────────
def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def bandpass(sig, fs, lo, hi, order=3):
    nyq  = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.asarray(sig, dtype=float))

def sig_marker(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*"   if p < 0.05  else "n.s.")

def format_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── DPF — Scholkmann & Wolf 2013 ───────────────────────────────────────────────────
def scholkmann_dpf(wl_nm, age):
    return (223.3
            + 0.05624   * (age   ** 0.8493)
            - 5.723e-7  * (wl_nm ** 3)
            + 0.001245  * (wl_nm ** 2)
            - 0.9025    *  wl_nm)


# ── OD helpers ─────────────────────────────────────────────────────────────────────
def _od(wl, pd_name):   return f"OD ({wl}nm) {pd_name}"
def _odp(wl, pd_name):  return f"ODp ({wl}nm) {pd_name}"

def _long_pds(df):
    """Accepted long PDs present in the file, restricted to TARGET_PDS (PD0/PD3 out)."""
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


# ── MBLL (long channels only) ───────────────────────────────────────────────────────
def run_mbll(df_odp, age):
    pds   = sorted({"PD" + c.split("PD")[1] for c in df_odp.columns
                    if c.startswith("ODp (") and "PD" in c})
    df_hb = pd.DataFrame()
    eHbO1, eHbR1 = EPS[WL1]
    eHbO2, eHbR2 = EPS[WL2]
    for pd_name in pds:
        c1, c2 = _odp(WL1, pd_name), _odp(WL2, pd_name)
        if c1 not in df_odp.columns or c2 not in df_odp.columns:
            continue
        sds = SDS.get(pd_name, 2.5)
        L1, L2 = sds * scholkmann_dpf(WL1, age), sds * scholkmann_dpf(WL2, age)
        A   = np.array([[eHbO1 * L1, eHbR1 * L1],
                        [eHbO2 * L2, eHbR2 * L2]])
        OD  = np.vstack([df_odp[c1].to_numpy(float), df_odp[c2].to_numpy(float)])
        C   = np.linalg.inv(A) @ OD
        df_hb[f"HbO_{pd_name}"] = C[0] * 1000.0
        df_hb[f"HbR_{pd_name}"] = C[1] * 1000.0
    return df_hb


# ── HRF / design ─────────────────────────────────────────────────────────────────
def spm_hrf(fs, length_s=32.0):
    dt = 1.0 / fs
    t  = np.arange(0, length_s, dt)
    def gamma_pdf(t, peak, disp):
        a = peak / disp
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = (a - 1) * np.log(t / disp) - (t / disp) - gammaln(a) - np.log(disp)
        p = np.exp(logp); p[~np.isfinite(p)] = 0.0
        return p
    h = gamma_pdf(t, 6.0, 1.0) - gamma_pdf(t, 16.0, 1.0) / 6.0
    return h / (np.max(np.abs(h)) + 1e-12)

def boxcar_regressor(t, onsets, durations, hrf):
    t   = np.asarray(t, dtype=float)
    dt  = np.median(np.diff(t))
    box = np.zeros(len(t), dtype=float)
    for onset, dur in zip(onsets, durations):
        i0 = max(0, min(np.searchsorted(t, onset),       len(t)))
        i1 = max(0, min(np.searchsorted(t, onset + dur), len(t)))
        if i1 > i0:
            box[i0:i1] = 1.0
    reg = np.convolve(box, hrf)[:len(t)] * dt
    return reg / (np.max(np.abs(reg)) + 1e-12)

def cosine_drift_basis(t, cutoff_s):
    N = len(t); T = t[-1] - t[0]
    K = int(np.floor(2 * T / cutoff_s))
    if K < 1:
        return np.empty((N, 0))
    n = np.arange(N); basis = np.zeros((N, K))
    for k in range(1, K + 1):
        basis[:, k - 1] = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return basis

def build_design(t, ev_2, ev_0, fs, extra=None):
    """Column order [task, *extra, drift_1..K, const]; task always at index 0."""
    hrf   = spm_hrf(fs)
    drift = cosine_drift_basis(t, DRIFT_CUTOFF_S)
    const = np.ones((len(t), 1))
    ev_all   = pd.concat([ev_2, ev_0], ignore_index=True).sort_values("onset")
    reg_task = boxcar_regressor(t, ev_all["onset"].to_numpy(),
                                ev_all["duration"].to_numpy(), hrf)
    regs, names = [reg_task], ["task"]
    for nm, arr in (extra or []):
        regs.append(np.asarray(arr, dtype=float)); names.append(nm)
    if drift.shape[1] > 0:
        X = np.column_stack(regs + [drift, const])
        names = names + [f"drift_{k+1}" for k in range(drift.shape[1])] + ["const"]
    else:
        X = np.column_stack(regs + [const]); names = names + ["const"]
    return X, names


# ── GLM ─────────────────────────────────────────────────────────────────────────────
def fit_glm(X, y):
    y       = np.asarray(y, dtype=float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas   = XtX_inv @ X.T @ y
    resid   = y - X @ betas
    r     = resid - resid.mean()
    denom = np.dot(r[:-1], r[:-1])
    rho   = float(np.clip(np.dot(r[1:], r[:-1]) / denom, -0.95, 0.95)) if denom > 0 else 0.0
    if abs(rho) > 1e-6:
        X_w, y_w = X.copy(), y.copy()
        X_w[1:] -= rho * X[:-1]; y_w[1:] -= rho * y[:-1]
        s0 = np.sqrt(1 - rho ** 2)
        X_w[0] *= s0; y_w[0] *= s0
        XtX_inv = np.linalg.pinv(X_w.T @ X_w)
        betas   = XtX_inv @ X_w.T @ y_w
        resid   = y_w - X_w @ betas
    df     = len(y) - X.shape[1]
    sigma2 = np.sum(resid ** 2) / df if df > 0 else np.nan
    return betas, XtX_inv, sigma2, df, rho

def contrast_test(betas, XtX_inv, sigma2, df, c):
    cb  = float(c @ betas)
    var = float(c @ XtX_inv @ c) * sigma2
    if var <= 0 or not np.isfinite(var):
        return cb, np.nan, np.nan
    se = np.sqrt(var)
    tval = cb / se
    pval = 2 * (1 - stats.t.cdf(abs(tval), df))
    return cb, se, pval

def compute_cnr(X, y, betas, task_idx):
    resid = y - X @ betas
    sigma = np.std(resid, ddof=X.shape[1])
    return abs(betas[task_idx]) / sigma if sigma > 1e-12 else np.nan

def fit_one(y, X, names):
    betas, XtX_inv, sigma2, df_resid, rho = fit_glm(X, y)
    idx = names.index("task")
    c   = np.zeros(X.shape[1]); c[idx] = 1.0
    cb, se, pval = contrast_test(betas, XtX_inv, sigma2, df_resid, c)
    cnr = compute_cnr(X, y, betas, idx)
    sigma_resid = float(np.std(y - X @ betas, ddof=X.shape[1]))
    coll = collinearity(X, names, idx)
    return dict(betas=betas, X=X, names=names, idx=idx,
                cb=cb, se=se, p=pval, rho=rho, cnr=cnr, sigma=sigma_resid, **coll)


def collinearity(X, names, task_idx):
    """
    How much the nuisance regressors overlap the task regressor.
      r_task_max : largest |corr| between task and any single nuisance
                   (which nuisance, in r_task_who) — flags PPG/SS eating task variance
      vif_task   : variance inflation factor of the task column = 1/(1-R²) where
                   R² is task regressed on all other columns. VIF≈1 ideal; >5 concerning.
    """
    task = X[:, task_idx]
    others_idx = [j for j in range(X.shape[1]) if j != task_idx and names[j] != "const"]
    # pairwise correlations with each NAMED nuisance (skip drift/const)
    r_max, who = 0.0, "-"
    for j in others_idx:
        nm = names[j]
        if nm.startswith("drift"):
            continue
        a, b = task - task.mean(), X[:, j] - X[:, j].mean()
        denom = np.sqrt((a @ a) * (b @ b))
        r = abs(a @ b) / denom if denom > 0 else 0.0
        if r > r_max:
            r_max, who = r, nm
    # VIF of task: regress task on all other columns
    if others_idx:
        Z = X[:, others_idx]
        beta_z = np.linalg.pinv(Z.T @ Z) @ Z.T @ task
        resid  = task - Z @ beta_z
        ss_tot = np.sum((task - task.mean()) ** 2)
        r2     = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
        vif    = 1.0 / (1.0 - r2) if r2 < 0.9999 else np.inf
    else:
        vif = 1.0
    return {"r_task_max": r_max, "r_task_who": who, "vif_task": vif}


# ── Data loading ──────────────────────────────────────────────────────────────────
def get_events(df, t):
    ev  = df["event_name"].astype(str).str.strip().to_numpy()
    idx = np.where(np.concatenate([[True], ev[1:] != ev[:-1]]))[0]
    o2, d2, o0, d0 = [], [], [], []
    for i, start in enumerate(idx):
        label = ev[start]
        end   = idx[i + 1] - 1 if i + 1 < len(idx) else len(t) - 1
        onset, dur = t[start], t[end] - t[start]
        if   label == "2-back": o2.append(onset); d2.append(dur)
        elif label == "0-back": o0.append(onset); d0.append(dur)
    return (pd.DataFrame({"onset": o2, "duration": d2}),
            pd.DataFrame({"onset": o0, "duration": d0}))

def get_all_onsets(ev_2, ev_0):
    return pd.concat([ev_2, ev_0]).sort_values("onset")["onset"].to_numpy()

def get_ss_od(df, n):
    """PD0 OD at 740 + 850 nm (already band-limited), z-scored. None if absent/flat."""
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
    dt   = np.median(np.diff(t))
    ep_t = np.arange(-n_pre, n_post) * dt
    epochs = []
    for onset in onsets:
        i_on = np.searchsorted(t, onset)
        i0, i1 = i_on - n_pre, i_on + n_post
        if i0 < 0 or i1 > len(y):
            continue
        ep = y[i0:i1]
        if len(ep) == len(ep_t):
            epochs.append(ep)
    return (np.array(epochs) if epochs else np.empty((0, len(ep_t)))), ep_t

def baseline_correct(epochs, ep_t):
    if epochs.size == 0:
        return epochs
    bl_mask = (ep_t >= BASELINE_FROM) & (ep_t < BASELINE_TO)
    return epochs - np.nanmean(epochs[:, bl_mask], axis=1, keepdims=True)

def mean_sem(epochs):
    mean = np.nanmean(epochs, axis=0)
    sem  = (np.nanstd(epochs, axis=0, ddof=1) / np.sqrt(epochs.shape[0])
            if epochs.shape[0] > 1 else np.zeros_like(mean))
    return mean, sem

def block_average(t_fit, onsets, hb_all, primary_res, n_min):
    """
    Per target PD, three block-averaged curves in the SAME space:
      - raw      : empirical HbO/HbR, no correction (mean ± SEM)
      - corrected: nuisance-regressed empirical = residual + task component,
                   i.e. y with SSR/PPG/drift removed (mean ± SEM)
      - glm      : GLM task component only = beta(task) × task regressor
    'corrected' and 'glm' live in the same corrected space; their difference is
    the unexplained residual. This visualizes exactly what the correction removes.
    """
    out = {}
    for pd_name in TARGET_PDS:
        pdr = {}
        for hb, key in [("HbO", "hbo"), ("HbR", "hbr")]:
            col = f"{hb}_{pd_name}"
            res = primary_res.get((pd_name, hb))
            if col not in hb_all.columns or res is None:
                continue
            y         = hb_all[col].to_numpy(float)[:n_min]
            X, betas  = res["X"], res["betas"]
            idx       = res["idx"]
            task_comp = betas[idx] * X[:, idx]           # GLM task component
            resid     = y - X @ betas                    # unexplained
            y_corr    = resid + task_comp                # nuisances + drift removed

            ep_t = None
            curves = {}
            for name, sig in [("raw", y), ("corr", y_corr), ("glm", task_comp)]:
                ep, ep_t = extract_epochs(t_fit, sig, onsets)
                ep       = baseline_correct(ep, ep_t)
                curves[name] = ep
            if curves["raw"].size == 0:
                continue
            m_raw,  s_raw  = mean_sem(curves["raw"])
            m_corr, s_corr = mean_sem(curves["corr"])
            m_glm,  _      = mean_sem(curves["glm"])
            pdr[f"{key}_raw_mean"]  = m_raw;  pdr[f"{key}_raw_sem"]  = s_raw
            pdr[f"{key}_corr_mean"] = m_corr; pdr[f"{key}_corr_sem"] = s_corr
            pdr[f"{key}_glm"]       = m_glm
            pdr["ep_t"] = ep_t; pdr["n_epochs"] = curves["raw"].shape[0]
        if pdr:
            out[pd_name] = pdr
    return out


# ── Per-subject plot: GLM fit overlay (p-values, no t) ──────────────────────────────
def plot_fits_cases(t, df_hb, results, case_keys, pd_name, ev_2, ev_0, subject):
    n  = min(len(t), len(df_hb)); tp = t[:n]
    cks = [c for c in case_keys if any((c, pd_name, hb) in results for hb in HB_TYPES)]
    if not cks:
        return
    fig, axes = plt.subplots(len(cks), 2, figsize=(14, 3.2 * len(cks)),
                             sharex=True, squeeze=False)
    for i, ck in enumerate(cks):
        for j, hb in enumerate(HB_TYPES):
            ax  = axes[i, j]
            res = results.get((ck, pd_name, hb)); col = f"{hb}_{pd_name}"
            if res is None or col not in df_hb.columns:
                ax.set_visible(False); continue
            y    = df_hb[col].to_numpy(float)[:n]
            comp = res["betas"][res["idx"]] * res["X"][:n, res["idx"]]
            for _, r in ev_2.iterrows():
                ax.axvspan(r["onset"], r["onset"] + r["duration"], color="#ff8c00", alpha=0.12, lw=0)
            for _, r in ev_0.iterrows():
                ax.axvspan(r["onset"], r["onset"] + r["duration"], color="#ffd700", alpha=0.12, lw=0)
            ax.plot(tp, y,                           lw=0.5, color="gray",  alpha=0.6, label="data")
            ax.plot(tp, res["X"][:n] @ res["betas"], lw=1.0, color="black", alpha=0.7, label="full fit")
            ax.plot(tp, comp, lw=2.0, color="#c0392b" if hb == "HbO" else "#2c5fa8",
                    label="task component")
            ax.axhline(0, color="k", lw=0.3)
            ax.set_title(f"{CASE_DEFS[ck]['label']}  |  {hb}  "
                         f"β={res['cb']:+.3f} µM  p={format_p(res['p'])}  {sig_marker(res['p'])}",
                         fontsize=9)
            ax.set_ylabel("Δ[Hb] (µM)"); ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"{subject} — {pd_name} — GLM fit by case (same data, different design)\n"
                 f"orange = 2-back, yellow = 0-back", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


# ── Per-subject plot: metric grouped bars ───────────────────────────────────────────
def plot_metric_grouped(results, case_keys, pds, subject, metric, ylabel, title, show_sig):
    chans = [(p, hb) for p in pds for hb in HB_TYPES
             if any((c, p, hb) in results for c in case_keys)]
    if not chans:
        return
    x = np.arange(len(chans)); width = 0.8 / len(case_keys)
    fig, ax = plt.subplots(figsize=(max(8, len(chans) * 2.0), 5))
    for k, ck in enumerate(case_keys):
        vals = [results.get((ck, *ch), {}).get(metric, np.nan) for ch in chans]
        errs = [results.get((ck, *ch), {}).get("se", 0) if metric == "cb" else 0 for ch in chans]
        errs = [e if np.isfinite(e) else 0 for e in errs]
        off  = (k - (len(case_keys) - 1) / 2) * width
        ax.bar(x + off, vals, width, yerr=errs if metric == "cb" else None, capsize=3,
               color=CASE_DEFS[ck]["color"], alpha=0.9, edgecolor="white", label=CASE_DEFS[ck]["label"])
        if show_sig:
            for xi, ch, v in zip(x, chans, vals):
                res = results.get((ck, *ch))
                if res is None or not np.isfinite(v):
                    continue
                s = sig_marker(res["p"])
                if s != "n.s.":
                    ax.text(xi + off, v, s, ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x)
    ax.set_xticklabels([f"{hb}\n{p}" for p, hb in chans], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{subject} — {title}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()


# ── Block-average + GLM-fit overlay grid ────────────────────────────────────────────
def plot_block_grid(ba_results, corr_of, pd_name):
    """
    Per subject: raw empirical (light dashed) vs nuisance-corrected empirical
    (bold solid + SEM) with the GLM task component (black) overlaid. Corrected
    empirical and GLM fit share the same space. Correction method (SSR / PPG-only)
    is shown in each title — dashing now means raw-vs-corrected, not subject type.
    """
    valid = [(s, r[pd_name]) for s, r in ba_results.items() if pd_name in r]
    if not valid:
        print(f"  No subjects with {pd_name} block average — skipping figure")
        return
    n = len(valid); ncols = min(NCOLS, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                             sharex=True, squeeze=False)
    axf = axes.flatten()
    for ax in axf[n:]:
        ax.set_visible(False)

    HBO_RAW, HBO_CORR = "#e8a0a0", "#c0392b"
    HBR_RAW, HBR_CORR = "#a8bfe0", "#2c5fa8"
    for ax, (subject, pdr) in zip(axf, valid):
        ep_t = pdr.get("ep_t", np.array([]))
        for key, c_raw, c_corr in [("hbo", HBO_RAW, HBO_CORR), ("hbr", HBR_RAW, HBR_CORR)]:
            if f"{key}_raw_mean" in pdr:
                ax.plot(ep_t, pdr[f"{key}_raw_mean"], color=c_raw, lw=1.0, ls="--",
                        alpha=0.9)
            if f"{key}_corr_mean" in pdr:
                ax.fill_between(ep_t, pdr[f"{key}_corr_mean"] - pdr[f"{key}_corr_sem"],
                                pdr[f"{key}_corr_mean"] + pdr[f"{key}_corr_sem"],
                                color=c_corr, alpha=0.18)
                ax.plot(ep_t, pdr[f"{key}_corr_mean"], color=c_corr, lw=1.9, ls="-")
        if "hbo_glm" in pdr:
            ax.plot(ep_t, pdr["hbo_glm"], color="black", lw=1.6, ls="-")
        if "hbr_glm" in pdr:
            ax.plot(ep_t, pdr["hbr_glm"], color="black", lw=1.6, ls="--")
        ax.axvspan(0, TASK_DUR_S, color="orange", alpha=0.08, lw=0)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.axhline(0, color="k", lw=0.3); ax.set_xlim(-PRE_S, POST_S)
        ax.set_title(f"{subject} — {corr_of.get(subject, '?')}",
                     fontsize=9, fontweight="bold")
        ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    for ax in axf[max(0, n - ncols):n]:
        ax.set_xlabel("Time relative to onset (s)", fontsize=8)
    for i in range(0, n, ncols):
        axf[i].set_ylabel("Δ[Hb] (µM)", fontsize=8)
    handles = [
        Line2D([0], [0], color="#c0392b", lw=1.0, ls="--", label="ΔHbO raw (uncorrected)"),
        Line2D([0], [0], color="#c0392b", lw=2.0, ls="-",  label="ΔHbO corrected"),
        Line2D([0], [0], color="#2c5fa8", lw=1.0, ls="--", label="ΔHbR raw (uncorrected)"),
        Line2D([0], [0], color="#2c5fa8", lw=2.0, ls="-",  label="ΔHbR corrected"),
        Line2D([0], [0], color="black",   lw=1.6, ls="-",  label="GLM fit HbO"),
        Line2D([0], [0], color="black",   lw=1.6, ls="--", label="GLM fit HbR"),
    ]
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle(f"Block-averaged HRF: raw vs corrected, with GLM fit — {pd_name}\n"
                 f"(title shows correction applied: SSR for Strand 1, PPG-only for Strand 2)",
                 fontsize=11, fontweight="bold", y=0.985)
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=8,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.92))
    plt.show()


# ── Group summary plot (matched / per-strand) ───────────────────────────────────────
def plot_group_summary(rows, case_keys, metric_col, ylabel, suptitle, subjects=None, note=""):
    df = pd.DataFrame(rows)
    df = df[df["case"].isin(case_keys)]
    if subjects is not None:
        df = df[df["Subject"].isin(subjects)]
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, hb in zip(axes, HB_TYPES):
        sub = df[df["Hb"] == hb]
        if sub.empty:
            ax.set_visible(False); continue
        pds = sorted(sub["PD"].unique()); x = np.arange(len(pds)); width = 0.8 / len(case_keys)
        ymax = sub[metric_col].abs().max()
        for k, ck in enumerate(case_keys):
            g    = sub[sub["case"] == ck].groupby("PD")[metric_col]
            mean = [g.mean().get(p, np.nan)  for p in pds]
            sem  = [g.sem().get(p, np.nan)   for p in pds]
            cnts = [int(g.count().get(p, 0)) for p in pds]
            off  = (k - (len(case_keys) - 1) / 2) * width
            ax.bar(x + off, mean, width, yerr=sem, capsize=3, color=CASE_DEFS[ck]["color"],
                   alpha=0.9, edgecolor="white", label=CASE_DEFS[ck]["label"])
            for xi, m, e, c in zip(x, mean, sem, cnts):
                if not np.isfinite(m):
                    continue
                yy = (m + (e if np.isfinite(e) else 0) if m >= 0
                      else m - (e if np.isfinite(e) else 0)) + np.sign(m if m else 1) * 0.04 * ymax
                ax.text(xi + off, yy, f"n={c}", ha="center",
                        va="bottom" if m >= 0 else "top", fontsize=6, color="#333")
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(pds, fontsize=10)
        ax.set_title(hb, fontsize=11, fontweight="bold"); ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=9, framealpha=0.85)
    fig.suptitle(f"{suptitle}   (mean ± SEM{', ' + note if note else ''})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


# ── Terminal tables ─────────────────────────────────────────────────────────────────
def print_case_table(rows, case_key):
    sub = [r for r in rows if r["case"] == case_key]
    if not sub:
        return
    df = pd.DataFrame(sub)
    disp = pd.DataFrame({
        "Subject": df["Subject"], "PD": df["PD"], "Hb": df["Hb"],
        "β(task) µM": df["beta_task"].map(lambda v: f"{v:+.4f}"),
        "SE":         df["SE"].map(lambda v: f"{v:.4f}"),
        "p":          df["p"].map(format_p),
        "sig":        df["sig"],
        "CNR":        df["CNR"].map(lambda v: f"{v:.4f}"),
        "σ(resid)":   df["sigma"].map(lambda v: f"{v:.4f}"),
        "AR1ρ":       df["AR1rho"].map(lambda v: f"{v:+.3f}"),
        "r(nuis·task)": df.apply(lambda r: f"{r['r_task']:.2f}({r['r_who']})", axis=1),
        "VIF":        df["vif"].map(lambda v: "inf" if not np.isfinite(v) else f"{v:.2f}"),
    })
    w = 110
    print(f"\n{'='*w}")
    print(f"  CASE — {CASE_DEFS[case_key]['label']}")
    print(f"{'='*w}")
    print(disp.to_string(index=False))
    print(f"{'='*w}")
    print("  sig: *p<0.05 **p<0.01 ***p<0.001 (uncorr.). t-values omitted (autocorrelation-inflated).")
    print("  r(nuis·task): max |corr| of any named nuisance with task regressor (which one).")
    print("  VIF: task-column variance inflation. r≳0.3 or VIF≳5 ⇒ nuisance is eating task")
    print("  variance → β(task) unstable (the PPG-only instability seen in Strand 1).")
    print(f"{'='*w}")

def print_comparison(rows, case_keys, value_key, value_label, fmt):
    df = pd.DataFrame([r for r in rows if r["case"] in case_keys])
    if df.empty:
        return
    piv = df.pivot_table(index=["Subject", "PD", "Hb"], columns="case",
                         values=value_key, aggfunc="first")
    piv = piv.reindex(columns=[c for c in case_keys if c in piv.columns])
    piv = piv.rename(columns={c: CASE_DEFS[c]["label"] for c in case_keys})
    w = 100
    print(f"\n{'='*w}")
    print(f"  COMPARISON — {value_label} across cases")
    print(f"{'='*w}")
    print(piv.to_string(float_format=fmt))
    print(f"{'='*w}")

def matched_subjects(rows, case_keys):
    df = pd.DataFrame([r for r in rows if r["case"] in case_keys])
    if df.empty:
        return []
    need = set(case_keys)
    by = df.groupby("Subject")["case"].apply(set)
    return sorted([s for s, cs in by.items() if need.issubset(cs)])

def print_group_means(rows, subjects, case_keys, metric_col, value_label, fmt):
    df = pd.DataFrame([r for r in rows if r["case"] in case_keys and r["Subject"] in subjects])
    if df.empty:
        return
    out = []
    for p in sorted(df["PD"].unique()):
        for hb in HB_TYPES:
            row = {"PD": p, "Hb": hb}
            for ck in case_keys:
                g = df[(df["PD"] == p) & (df["Hb"] == hb) & (df["case"] == ck)][metric_col]
                row[CASE_DEFS[ck]["label"]] = f"{fmt(g.mean())} ± {fmt(g.sem())}" if len(g) else "—"
            out.append(row)
    w = 100
    print(f"\n{'='*w}")
    print(f"  GROUP MEAN ± SEM — {value_label}   [matched subjects, n={len(subjects)}]")
    print(f"{'='*w}")
    print(pd.DataFrame(out).to_string(index=False))
    print(f"{'='*w}")


# ── Grand-average HRF across subjects (headline task figure) ────────────────────────
def plot_grand_average(ba_results, strand_of, strand_val, corr_label):
    """
    Group grand-average block-averaged HRF (mean ± SEM across subjects), per PD.
    Shows raw (light dashed) and corrected (bold + SEM band) for HbO and HbR.
    Restricted to one strand so the correction is uniform.
    """
    subs = [s for s in ba_results if strand_of.get(s) == strand_val]
    if not subs:
        return
    pds = [p for p in TARGET_PDS if any(p in ba_results[s] for s in subs)]
    if not pds:
        return
    fig, axes = plt.subplots(1, len(pds), figsize=(6.5 * len(pds), 4.8), squeeze=False)
    for ax, pd_name in zip(axes[0], pds):
        ssubs = [s for s in subs if pd_name in ba_results[s]]
        ep_t  = ba_results[ssubs[0]][pd_name]["ep_t"]
        n     = len(ssubs)
        for key, c_raw, c_corr, lbl in [("hbo", "#e8a0a0", "#c0392b", "HbO"),
                                        ("hbr", "#a8bfe0", "#2c5fa8", "HbR")]:
            raw  = np.vstack([ba_results[s][pd_name][f"{key}_raw_mean"]  for s in ssubs])
            corr = np.vstack([ba_results[s][pd_name][f"{key}_corr_mean"] for s in ssubs])
            gm_raw  = raw.mean(0)
            gm_corr = corr.mean(0)
            sem_corr = corr.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(gm_corr)
            ax.plot(ep_t, gm_raw, color=c_raw, lw=1.0, ls="--", alpha=0.9)
            ax.fill_between(ep_t, gm_corr - sem_corr, gm_corr + sem_corr, color=c_corr, alpha=0.18)
            ax.plot(ep_t, gm_corr, color=c_corr, lw=2.2, label=f"Δ{lbl} corrected")
        ax.axvspan(0, TASK_DUR_S, color="orange", alpha=0.08, lw=0)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4); ax.axhline(0, color="k", lw=0.3)
        ax.set_xlim(-PRE_S, POST_S); ax.set_xlabel("Time relative to onset (s)")
        ax.set_ylabel("Δ[Hb] (µM)")
        ax.set_title(f"{pd_name}  (n={n})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    handles = [Line2D([0], [0], color="#888", lw=1.0, ls="--", label="raw (uncorrected)"),
               Line2D([0], [0], color="#888", lw=2.2, label="corrected")]
    axes[0][0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle(f"Grand-average block-averaged HRF — {corr_label}-corrected "
                 f"(mean ± SEM across subjects)", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


def _peak_hbo(ba_results, subject, pd_name):
    """Peak corrected ΔHbO during task window and its latency, from block average."""
    pdr = ba_results.get(subject, {}).get(pd_name)
    if pdr is None or "hbo_corr_mean" not in pdr:
        return None, None
    ep_t = pdr["ep_t"]; y = pdr["hbo_corr_mean"]
    mask = (ep_t >= 0) & (ep_t <= TASK_DUR_S)
    if not mask.any():
        return None, None
    seg = y[mask]; ts = ep_t[mask]
    i = int(np.argmax(seg))
    return float(seg[i]), float(ts[i])


# ── Abstract-ready summary numbers ──────────────────────────────────────────────────
def print_abstract_summary(rows, ba_results, strand, case_key, label):
    """Detection rate, group-mean β/CNR, and block-average peak ΔHbO — per channel."""
    sub = [r for r in rows if r["strand"] == strand and r["case"] == case_key]
    if not sub:
        return
    df = pd.DataFrame(sub)
    w = 104
    print(f"\n{'='*w}")
    print(f"  ABSTRACT SUMMARY — Strand {strand} ({label})   [detection: HbO β>0 & p<0.05]")
    print(f"{'='*w}")
    out = []
    for pd_name in TARGET_PDS:
        hbo = df[(df["PD"] == pd_name) & (df["Hb"] == "HbO")]
        hbr = df[(df["PD"] == pd_name) & (df["Hb"] == "HbR")]
        if hbo.empty:
            continue
        subs = sorted(hbo["Subject"].unique())
        n = len(subs)
        det_A = int(((hbo["beta_task"] > 0) & (hbo["p"] < 0.05)).sum())
        # stricter: HbO up (p<.05) AND HbR down, same subject
        det_B = 0
        for s in subs:
            ho = hbo[hbo["Subject"] == s]; hr = hbr[hbr["Subject"] == s]
            if len(ho) and len(hr):
                if (ho["beta_task"].iloc[0] > 0 and ho["p"].iloc[0] < 0.05
                        and hr["beta_task"].iloc[0] < 0):
                    det_B += 1
        peaks = [p for p, _ in (_peak_hbo(ba_results, s, pd_name) for s in subs) if p is not None]
        lats  = [l for _, l in (_peak_hbo(ba_results, s, pd_name) for s in subs) if l is not None]
        out.append({
            "PD": pd_name, "n": n,
            "detect HbO↑": f"{det_A}/{n}",
            "detect HbO↑&HbR↓": f"{det_B}/{n}",
            "β HbO µM (m±sem)": f"{hbo['beta_task'].mean():+.3f}±{hbo['beta_task'].sem():.3f}",
            "β HbR µM (m±sem)": f"{hbr['beta_task'].mean():+.3f}±{hbr['beta_task'].sem():.3f}",
            "CNR HbO (m±sem)":  f"{hbo['CNR'].mean():.3f}±{hbo['CNR'].sem():.3f}",
            "peak ΔHbO µM":     (f"{np.mean(peaks):+.3f}±{np.std(peaks, ddof=1)/np.sqrt(len(peaks)):.3f}"
                                 if len(peaks) > 1 else (f"{peaks[0]:+.3f}" if peaks else "—")),
            "t-peak s":         (f"{np.mean(lats):.1f}" if lats else "—"),
        })
    print(pd.DataFrame(out).to_string(index=False))
    print(f"{'='*w}")
    print("  Report n with every value. Peak ΔHbO from SSR/PPG-corrected block average,")
    print("  task window 0–{:.0f}s. Amplitudes approximate (custom-sensor pathlength).".format(TASK_DUR_S))
    print(f"{'='*w}")


# ── SSR effect: residual-noise reduction + coupling recovery ────────────────────────
def print_ssr_effect(rows):
    """Strand-1 only: σ(residual) reduction no_ssr→ssr, and HbO↑/HbR↓ coupling recovery."""
    df = pd.DataFrame([r for r in rows if r["strand"] == 1])
    if df.empty:
        return
    w = 104
    print(f"\n{'='*w}")
    print(f"  SSR EFFECT (Strand 1) — residual-noise reduction & coupling recovery")
    print(f"{'='*w}")
    # σ reduction per channel (paired no_ssr → ssr)
    sig_out = []
    for pd_name in TARGET_PDS:
        for hb in HB_TYPES:
            piv = df[(df["PD"] == pd_name) & (df["Hb"] == hb)].pivot_table(
                index="Subject", columns="case", values="sigma", aggfunc="first")
            if "no_ssr" not in piv or "ssr" not in piv:
                continue
            paired = piv[["no_ssr", "ssr"]].dropna()
            if paired.empty:
                continue
            red = (paired["no_ssr"] - paired["ssr"]) / paired["no_ssr"] * 100.0
            sig_out.append({"PD": pd_name, "Hb": hb, "n": len(paired),
                            "median σ reduction %": f"{red.median():+.1f}",
                            "mean σ reduction %":   f"{red.mean():+.1f}"})
    if sig_out:
        print("\n  Residual-noise (σ) change after SSR  [positive = SSR removed noise]")
        print(pd.DataFrame(sig_out).to_string(index=False))
    # coupling recovery per (subject, PD): HbO>0 & HbR<0
    coup = []
    for ck in ["no_ssr", "ssr"]:
        ok = tot = 0
        for pd_name in TARGET_PDS:
            d = df[(df["case"] == ck) & (df["PD"] == pd_name)]
            for s in d["Subject"].unique():
                ho = d[(d["Subject"] == s) & (d["Hb"] == "HbO")]
                hr = d[(d["Subject"] == s) & (d["Hb"] == "HbR")]
                if len(ho) and len(hr):
                    tot += 1
                    if ho["beta_task"].iloc[0] > 0 and hr["beta_task"].iloc[0] < 0:
                        ok += 1
        coup.append({"case": CASE_DEFS[ck]["label"],
                     "correct HbO↑/HbR↓": f"{ok}/{tot}",
                     "fraction": f"{ok/tot:.2f}" if tot else "—"})
    print("\n  Canonical coupling (HbO↑ & HbR↓) channel count, before vs after SSR")
    print(pd.DataFrame(coup).to_string(index=False))
    print(f"{'='*w}")


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    subject_folders = sorted([
        os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    # Anonymized subject IDs (stable alphabetical order) — real names never printed/plotted.
    subj_ids = {os.path.basename(f): f"S{i+1:02d}" for i, f in enumerate(subject_folders)}

    print("\nStep 9: GLM systemic-correction evaluation (PD1, PD2 only; PD3 excluded)")
    print("Strand 1 (PD0 survived): no_ssr / ssr / ssr+ppg   |   "
          "Strand 2 (PD0 dropped): no_ssr / ppg_only")
    print("Reporting beta, SE, p (t omitted), CNR.  Contrast: HbO>0, HbR<0 expected\n")

    all_rows    = []     # GLM rows: per (subject, case, PD, Hb)
    ba_results  = {}     # block-average curves: subject → {PD → {...}}
    strand_of   = {}     # subject → 1 or 2
    n_plotted   = 0

    for folder in subject_folders:
        subject = os.path.basename(folder)
        sid     = subj_ids[subject]
        od_path = os.path.join(folder, f"{subject}_preprocessed_OD.csv")
        if not os.path.exists(od_path):
            print(f"  {sid}: no preprocessed OD file — skipping"); continue
        age = SUBJECT_AGES.get(subject.lower())
        if age is None:
            print(f"  {sid}: age missing — skipping"); continue

        try:
            df = pd.read_csv(od_path, low_memory=False)
            if "time_s" not in df.columns:
                df["time_s"] = np.arange(len(df)) / FS
            t  = df["time_s"].to_numpy(float)
            fs = 1.0 / np.median(np.diff(t))
            if "event_name" not in df.columns:
                print(f"  {sid}: no event_name — skipping"); continue
            ev_2, ev_0 = get_events(df, t)
            if ev_2.empty or ev_0.empty:
                print(f"  {sid}: no task blocks — skipping"); continue
            onsets = get_all_onsets(ev_2, ev_0)

            hb_all = run_mbll(long_od_frame(df), age)
            long_pds = sorted([c.replace("HbO_", "") for c in hb_all.columns
                               if c.startswith("HbO_")])
            if not long_pds:
                print(f"  {sid}: no usable long channels (PD1/PD2) — skipping"); continue

            ss_regs = get_ss_od(df, len(t))
            ppg_reg = get_ppg(df, len(t))
            has_ss, has_ppg = ss_regs is not None, ppg_reg is not None

            # Route to a strand and pick which cases to run
            strand     = 1 if has_ss else 2
            strand_of[sid] = strand
            case_keys  = STRAND1_CASES if has_ss else STRAND2_CASES

            X0, _ = build_design(t, ev_2, ev_0, fs, extra=None)
            n_min = min(len(t), X0.shape[0])

            results = {}
            for ck in case_keys:
                cd = CASE_DEFS[ck]
                if cd["ssr"] and not has_ss:  continue
                if cd["ppg"] and not has_ppg: continue
                extra = []
                if cd["ssr"]: extra += ss_regs
                if cd["ppg"]: extra.append(("PPG", ppg_reg))
                X, names = build_design(t, ev_2, ev_0, fs, extra)
                Xf = X[:n_min]
                for pd_name in long_pds:
                    for hb in HB_TYPES:
                        col = f"{hb}_{pd_name}"
                        if col not in hb_all.columns: continue
                        y   = hb_all[col].to_numpy(float)[:n_min]
                        res = fit_one(y, Xf, names)
                        results[(ck, pd_name, hb)] = res
                        all_rows.append({
                            "Subject": sid, "strand": strand, "case": ck,
                            "PD": pd_name, "Hb": hb, "beta_task": res["cb"],
                            "SE": res["se"], "p": res["p"], "sig": sig_marker(res["p"]),
                            "CNR": res["cnr"], "AR1rho": res["rho"],
                            "sigma": res["sigma"],
                            "r_task": res["r_task_max"], "r_who": res["r_task_who"],
                            "vif": res["vif_task"],
                        })

            # Block average: overlay task component from the primary corrected case
            primary_ck  = "ssr" if has_ss else ("ppg_only" if has_ppg else "no_ssr")
            primary_res = {(p, hb): results[(primary_ck, p, hb)]
                           for p in long_pds for hb in HB_TYPES
                           if (primary_ck, p, hb) in results}
            ba = block_average(t[:n_min], onsets, hb_all, primary_res, n_min)
            if ba:
                ba_results[sid] = ba

            tag = (f"STRAND 1 [SSR{'+PPG' if has_ppg else ''}]" if has_ss
                   else f"STRAND 2 [{'PPG-only' if has_ppg else 'no correction'}]")
            print(f"  {sid}: {long_pds}, {len(ev_2)}×2-back + {len(ev_0)}×0-back  → {tag}")

            if n_plotted < N_PLOT:
                rep_pd = "PD1" if "PD1" in long_pds else long_pds[0]
                plot_fits_cases(t, hb_all, results, case_keys, rep_pd, ev_2, ev_0, sid)
                plot_metric_grouped(results, case_keys, long_pds, sid, "cb",
                                    "β(task) (µM)", "β(task) by case", show_sig=True)
                # CNR is still computed/reported (all_rows, print_* tables) — just not plotted
                # plot_metric_grouped(results, case_keys, long_pds, sid, "cnr",
                #                     "CNR = |β(task)| / σ(resid)", "CNR by case", show_sig=False)
                n_plotted += 1
        except Exception as e:
            import traceback
            print(f"\n  ERROR — {sid}: {e}"); traceback.print_exc()

    if not all_rows:
        print("\nNo results produced."); return

    s1 = sorted([s for s, st in strand_of.items() if st == 1])
    s2 = sorted([s for s, st in strand_of.items() if st == 2])
    w  = 100
    print(f"\n{'#'*w}\n#  STRAND ASSIGNMENT\n{'#'*w}")
    print(f"  Strand 1 (PD0 survived → SSR), n={len(s1)}: {s1}")
    print(f"  Strand 2 (PD0 dropped → PPG-only), n={len(s2)}: {s2}")

    # ════════════════ STRAND 1 ════════════════
    print(f"\n{'#'*w}\n#  STRAND 1 — short-separation regression evaluation\n{'#'*w}")
    for ck in STRAND1_CASES:
        print_case_table([r for r in all_rows if r["strand"] == 1], ck)
    print_comparison([r for r in all_rows if r["strand"] == 1], STRAND1_CASES,
                     "beta_task", "β(task) (µM)", lambda x: f"{x:+.4f}")
    print_comparison([r for r in all_rows if r["strand"] == 1], STRAND1_CASES,
                     "CNR", "CNR", lambda x: f"{x:.4f}")
    matched1 = matched_subjects([r for r in all_rows if r["strand"] == 1], STRAND1_CASES)
    print(f"\n  Matched (all 3 cases present), n={len(matched1)}: {matched1}")
    print_group_means(all_rows, matched1, STRAND1_CASES, "beta_task",
                      "β(task) (µM)", lambda x: f"{x:+.4f}")
    print_group_means(all_rows, matched1, STRAND1_CASES, "CNR", "CNR", lambda x: f"{x:.4f}")
    plot_group_summary(all_rows, STRAND1_CASES, "beta_task", "β(task) (µM)",
                       "Strand 1: group-mean β(task), by case", subjects=matched1,
                       note=f"matched n={len(matched1)}")
    # CNR is still computed/reported (print_comparison/print_group_means above) — just not plotted
    # plot_group_summary(all_rows, STRAND1_CASES, "CNR", "CNR",
    #                    "Strand 1: group-mean CNR, by case", subjects=matched1,
    #                    note=f"matched n={len(matched1)}")
    print_ssr_effect(all_rows)
    print_abstract_summary(all_rows, ba_results, 1, "ssr", "SSR-corrected")

    # ════════════════ STRAND 2 ════════════════
    if s2:
        print(f"\n{'#'*w}\n#  STRAND 2 — PPG-only (no short channel; best available correction)\n{'#'*w}")
        for ck in STRAND2_CASES:
            print_case_table([r for r in all_rows if r["strand"] == 2], ck)
        print_comparison([r for r in all_rows if r["strand"] == 2], STRAND2_CASES,
                         "beta_task", "β(task) (µM)", lambda x: f"{x:+.4f}")
        print_comparison([r for r in all_rows if r["strand"] == 2], STRAND2_CASES,
                         "CNR", "CNR", lambda x: f"{x:.4f}")
        matched2 = matched_subjects([r for r in all_rows if r["strand"] == 2], STRAND2_CASES)
        print(f"\n  Matched (both cases present), n={len(matched2)}: {matched2}")
        plot_group_summary(all_rows, STRAND2_CASES, "beta_task", "β(task) (µM)",
                           "Strand 2: group-mean β(task), by case", subjects=matched2,
                           note=f"PPG-only fallback, matched n={len(matched2)}")
        # CNR is still computed/reported (print_comparison above) — just not plotted
        # plot_group_summary(all_rows, STRAND2_CASES, "CNR", "CNR",
        #                    "Strand 2: group-mean CNR, by case", subjects=matched2,
        #                    note=f"PPG-only fallback, matched n={len(matched2)}")
        print_abstract_summary(all_rows, ba_results, 2, "ppg_only", "PPG-only-corrected")

    # ════════════════ BLOCK AVERAGE (all subjects, both strands) ════════════════
    if ba_results:
        corr_of = {s: ("SSR" if strand_of.get(s) == 1 else "PPG-only") for s in ba_results}
        print(f"\n{'#'*w}\n#  BLOCK-AVERAGED HRF: raw vs corrected + GLM fit\n{'#'*w}")
        for pd_name in TARGET_PDS:
            n_subj = sum(1 for r in ba_results.values() if pd_name in r)
            print(f"\n  Plotting block-average grid — {pd_name} ({n_subj} subject(s)) ...")
            plot_block_grid(ba_results, corr_of, pd_name)

        # Headline group figure: grand-average HRF per strand
        print("\n  Plotting grand-average HRF (Strand 1, SSR-corrected) ...")
        plot_grand_average(ba_results, strand_of, 1, "SSR")
        if s2:
            print("  Plotting grand-average HRF (Strand 2, PPG-only-corrected) ...")
            plot_grand_average(ba_results, strand_of, 2, "PPG-only")


if __name__ == "__main__":
    main()