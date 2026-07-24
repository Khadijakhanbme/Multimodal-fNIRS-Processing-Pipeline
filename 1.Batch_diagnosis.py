"""
batch_diagnostic.py
===================
Batch fNIRS raw signal diagnostic across all subjects.

Usage:
    python batch_diagnostic.py --data_dir path/to/rawdata/2nd_Trial

Expects folder structure:
    2nd_Trial/
        Esin_custom/
            Esin_signals.csv   ← naming pattern: {Subject}_signals.csv
        Lina_custom/
            Lina_signals.csv
        ...

Prints a formatted summary table + per-subject issues list.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import welch

# ── thresholds ────────────────────────────────────────────────────────────────
TARGET_FS         = 20.0
GAP_WARN_PCT      = 5.0    # % → caution
GAP_FAIL_PCT      = 15.0   # % → do not process
TRUE_DISCONNECT_S = 1.0    # s → true disconnection
SIGNAL_MIN_V      = 0.02   # V — low-power LED sensor (not research-grade)
COUPLING_STEP_V   = 0.05   # V — coupling shift threshold
MAYER_DOMINANT    = 5.0    # ratio — Mayer/cardiac dominance threshold
PHOTODIODES       = (0, 1, 2, 3)

# ── find raw file ─────────────────────────────────────────────────────────────
def find_raw_csv(folder):
    """Find the raw signals CSV — named {Subject}_signals.csv"""
    for f in sorted(os.listdir(folder)):
        fl = f.lower()
        if (f.endswith(".csv")
                and "signals" in fl
                and "resampled" not in fl
                and "events"    not in fl
                and "aligned"   not in fl
                and "quality"   not in fl):
            return os.path.join(folder, f)
    # fallback: any CSV not matching pipeline outputs
    for f in sorted(os.listdir(folder)):
        fl = f.lower()
        if (f.endswith(".csv")
                and "resampled" not in fl
                and "events"    not in fl
                and "aligned"   not in fl
                and "quality"   not in fl
                and "diagnostic" not in fl):
            return os.path.join(folder, f)
    return None

# ── diagnose one subject ──────────────────────────────────────────────────────
def diagnose(subject, raw_path):
    r = dict(subject=subject, duration_min=None, gap_pct=None,
             true_disc=0, burst_pct=None, os_jitter_pct=None,
             tia_ppg=False, tia_fnirs=False, coupling=[],
             sig_pd0=None, sig_pd1=None, sig_pd2=None, sig_pd3=None,
             mayer_pd0=None, mayer_pd1=None,
             issues=[], warnings=[], verdict=None)
    try:
        df = pd.read_csv(raw_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df = df.sort_values('timestamp').reset_index(drop=True)
        diffs = df['timestamp'].diff().dt.total_seconds().dropna()

        duration  = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        fs_est    = 1.0 / diffs.median()
        burst_pct = (diffs < 0.01).sum() / len(diffs) * 100

        r["duration_min"] = round(duration / 60, 1)
        r["burst_pct"]    = round(burst_pct, 1)

        # ── gaps ──
        gaps      = diffs[diffs > 3.0/TARGET_FS]
        true_disc = diffs[diffs > TRUE_DISCONNECT_S]
        gap_pct   = gaps.sum() / duration * 100
        burst_idx = set(diffs[diffs < 0.01].index)
        os_pct    = sum(1 for g in gaps.index
                        if any((g+i) in burst_idx for i in range(1,10))) \
                    / max(1, len(gaps)) * 100

        r["gap_pct"]      = round(gap_pct, 1)
        r["true_disc"]    = len(true_disc)
        r["os_jitter_pct"]= round(os_pct, 1)

        if len(true_disc) > 0:
            r["issues"].append(f"{len(true_disc)} true disconnection(s)")
        if gap_pct >= GAP_FAIL_PCT:
            r["issues"].append(f"excessive gaps ({gap_pct:.1f}%)")
        elif gap_pct >= GAP_WARN_PCT:
            r["warnings"].append(f"moderate gaps ({gap_pct:.1f}%)")
        if burst_pct > 40:
            r["warnings"].append(f"high burst rate ({burst_pct:.1f}%)")
        elif burst_pct > 20:
            r["warnings"].append(f"moderate burst rate ({burst_pct:.1f}%)")

        # ── TIA switches ──

        # Only check PPG channels and analysis wavelengths (740nm, 850nm)
        relevant_tia = [c for c in df.columns if 'TIA' in c and 
                any(wl in c for wl in ['465nm', '515nm', '635nm', '740nm', '850nm'])]
        for col in relevant_tia:
            gain = df[col].to_numpy(float)
            if int(np.sum(np.diff(gain) != 0)) > 0:
                wl = col.replace("TIA gain (","").replace(")","")
                is_ppg = int(wl.replace("nm","")) < 700
                if is_ppg:
                    r["tia_ppg"] = True
                    r["issues"].append(f"PPG TIA switch ({wl})")
                else:
                    r["tia_fnirs"] = True                              # ← add this

                    r["warnings"].append(f"fNIRS TIA switch ({wl})")

        # ── coupling shifts ──
        for pd_idx in PHOTODIODES:
            col = f"fNIRS (810nm) PD{pd_idx}"
            if col not in df.columns: continue
            sig = df[col].to_numpy(float)
            big = np.where(np.abs(np.diff(sig)) > COUPLING_STEP_V)[0]
            if len(big) > 0:
                t_s = (df['timestamp'].iloc[big[0]] - df['timestamp'].iloc[0]).total_seconds()
                r["coupling"].append(f"PD{pd_idx}@{t_s:.0f}s")
                r["warnings"].append(f"PD{pd_idx} coupling shift @{t_s:.0f}s")

        # ── signal presence — just is there signal? ──
        for pd_idx in PHOTODIODES:
            c740 = f"fNIRS (740nm) PD{pd_idx}"
            c850 = f"fNIRS (850nm) PD{pd_idx}"
            if c740 not in df.columns: continue
            m = min(df[c740].mean(), df[c850].mean())
            key = f"sig_pd{pd_idx}"
            if m >= SIGNAL_MIN_V * 2:
                r[key] = "OK"
            elif m >= SIGNAL_MIN_V:
                r[key] = "weak"
                r["warnings"].append(f"PD{pd_idx} weak signal ({m:.3f}V)")
            else:
                r[key] = "none"
                # Only critical if PD0 has no signal (sensor completely off)
                if pd_idx == 0:
                    r["issues"].append("PD0 no signal — sensor off scalp?")
                else:
                    r["warnings"].append(f"PD{pd_idx} no signal (expected at long sep)")

        # ── Mayer ratio ──
        t_raw = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds().to_numpy()
        t_uni = np.arange(0, t_raw[-1], 1.0/TARGET_FS)
        for pd_idx in [0, 1]:
            col = f"fNIRS (740nm) PD{pd_idx}"
            if col not in df.columns: continue
            sig_r = np.interp(t_uni, t_raw, df[col].to_numpy(float))
            f, pxx = welch(sig_r, fs=TARGET_FS, nperseg=512, noverlap=256, detrend="linear")
            mp = np.mean(pxx[(f >= 0.08) & (f <= 0.15)])
            cp = np.mean(pxx[(f >= 1.0)  & (f <= 2.0)])
            ratio = round(mp / cp, 1) if cp > 0 else 999
            r[f"mayer_pd{pd_idx}"] = ratio
            if ratio > MAYER_DOMINANT:
                r["warnings"].append(f"PD{pd_idx} Mayer dominant ({ratio}x)")

        # ── verdict ──
        if r["issues"]:
            r["verdict"] = "CAUTION" if not any(
                "excessive gaps" in i or "sensor off scalp" in i
                for i in r["issues"]) else "DO NOT PROCESS"
        else:
            r["verdict"] = "PROCEED"

    except Exception as e:
        r["verdict"] = "ERROR"
        r["issues"].append(str(e))
    return r

# ── table ─────────────────────────────────────────────────────────────────────
def sig_icon(v):
    return {"OK": "✅", "weak": "⚠️ ", "none": "❌", None: "--"}.get(v, "--")

def mayer_str(v):
    if v is None: return "    --  "
    return f"{v:6.1f}x"

def print_table(results):
    # Assign anonymous IDs in alphabetical order
    all_subj = sorted(r["subject"] for r in results)
    subj_id  = {s: f"S{i+1:02d}" for i, s in enumerate(all_subj)}

    W = 101
    print(f"\n{'═'*W}")
    print(f"  fNIRS BATCH RAW DIAGNOSTIC")
    print(f"{'═'*W}")

    h = (f"  {'ID':<5}  {'Min':>4}  {'Gap%':>5}  {'Disc':>4}  "
         f"{'Burst%':>6}  {'OSJit%':>6}  {'PPG':>3}  {'fNIRS':>5}  {'Coupling':<12}  "
         f"{'PD0':>3} {'PD1':>3} {'PD2':>3} {'PD3':>3}  "
         f"{'Mayer0':>8} {'Mayer1':>8}")
    print(h)
    print(f"  {'─'*97}")

    for r in results:
        sid       = subj_id[r["subject"]]
        tia_ppg   = "❌" if r["tia_ppg"]   else "✅"
        tia_fnirs = "❌" if r["tia_fnirs"] else "✅"
        coup = ",".join(r["coupling"]) if r["coupling"] else "none"
        coup = coup[:12]

        row = (f"  {sid:<5}  "
               f"{str(r['duration_min'] or '--'):>4}  "
               f"{str(r['gap_pct'] or '--'):>5}  "
               f"{str(r['true_disc']):>4}  "
               f"{str(r['burst_pct'] or '--'):>6}  "
               f"{str(r['os_jitter_pct'] or '--'):>6}  "
               f"{tia_ppg:>3}  {tia_fnirs:>5}  "
               f"{coup:<12}  "
               f"{sig_icon(r['sig_pd0'])} "
               f"{sig_icon(r['sig_pd1'])} "
               f"{sig_icon(r['sig_pd2'])} "
               f"{sig_icon(r['sig_pd3'])}  "
               f"{mayer_str(r['mayer_pd0'])} "
               f"{mayer_str(r['mayer_pd1'])}")
        print(row)

    print(f"{'═'*W}")
    print(f"  ID=anonymous subject ID  Min=duration(min)  Gap%=gap rate  Disc=true disconnections")
    print(f"  Burst%=jitter bursts  OSJit%=OS-jitter confirmed")
    print(f"  PPG/fNIRS=TIA gain stable  PD0-3=signal (✅OK  ⚠️weak  ❌none)")
    print(f"  Mayer0/1 = Mayer-wave / cardiac power ratio for PD0 / PD1  (>5x = dominant)")
    print(f"{'═'*W}\n")

# ── issues ────────────────────────────────────────────────────────────────────
def print_issues(results):
    has_any = any(r["issues"] or r["warnings"] for r in results)
    if not has_any:
        print("  All subjects clean — no issues or warnings.\n")
        return

    print(f"\n  ISSUES & WARNINGS")
    print(f"  {'─'*55}")
    for r in results:
        if not r["issues"] and not r["warnings"]:
            continue
        print(f"\n  {r['subject']}  [{r['verdict']}]")
        for i in r["issues"]:
            print(f"    ❌  {i}")
        for w in r["warnings"]:
            print(f"    ⚠️   {w}")
    print()

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # Hardcoded path to rawdata
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "rawdata", "2nd Trial", "Multimodal_data")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    subfolders = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not subfolders:
        print(f"No subfolders found in: {data_dir}")
        sys.exit(1)

    print(f"\nScanning {len(subfolders)} subject folder(s)...")
    results = []
    for folder in subfolders:
        subject  = os.path.basename(folder).replace("_custom","")
        raw_path = find_raw_csv(folder)
        if raw_path is None:
            print(f"  ⚠️  No raw CSV in {folder} — skipping")
            continue
        print(f"  {subject}...")
        results.append(diagnose(subject, raw_path))

    if results:
        print_table(results)
        print_issues(results)

if __name__ == "__main__":
    main()