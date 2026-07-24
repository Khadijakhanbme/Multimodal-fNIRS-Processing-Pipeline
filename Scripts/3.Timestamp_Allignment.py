#!/usr/bin/env python3
import os
import csv
import sys
from datetime import datetime
from pathlib import Path


def parse_iso_timestamp(ts_str):
    """Convert ISO format timestamp to datetime object."""
    try:
        return datetime.fromisoformat(ts_str)
    except:
        return None


def load_task_events(events_file):
    """
    Load task events and extract all periods (task, rest, baseline).
    """
    periods = []
    events = {}

    with open(events_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_num  = int(row['block_number'])
            event_type = row['event_type'].strip()
            ts_wall    = parse_iso_timestamp(row['timestamp_wall_clock'])

            if block_num not in events:
                events[block_num] = {}

            if event_type == 'cue_start':
                events[block_num]['cue_start']  = ts_wall
                events[block_num]['condition']  = row['condition'].strip()
            elif event_type == 'task_end':
                events[block_num]['task_end']   = ts_wall
            elif event_type == 'rest_start':
                events[block_num]['rest_start'] = ts_wall
            elif event_type == 'rest_end':
                events[block_num]['rest_end']   = ts_wall
            elif event_type == 'baseline_start':
                events[block_num]['baseline_start'] = ts_wall

    # Build periods list: task blocks and rest blocks
    for block_num in sorted(events.keys()):
        block_data = events[block_num]

        if 'cue_start' in block_data and 'task_end' in block_data:
            periods.append({
                'start':      block_data['cue_start'],
                'end':        block_data['task_end'],
                'event':      'task',
                'event_name': block_data.get('condition', 'unknown')
            })

        if 'rest_start' in block_data and 'rest_end' in block_data:
            periods.append({
                'start':      block_data['rest_start'],
                'end':        block_data['rest_end'],
                'event':      'rest',
                'event_name': 'rest'
            })

    # Add baseline
    if 0 in events and 'baseline_start' in events[0]:
        first_task_start = min([p['start'] for p in periods if p['event'] == 'task'])
        periods.append({
            'start':      events[0]['baseline_start'],
            'end':        first_task_start,
            'event':      'baseline',
            'event_name': 'baseline'
        })

    periods.sort(key=lambda x: x['start'])
    return periods


def get_experiment_window(periods):
    """
    Return (exp_start, exp_end) — the earliest period start and latest
    period end.  Everything outside this window gets discarded.
    """
    exp_start = min(p['start'] for p in periods)
    exp_end   = max(p['end']   for p in periods)
    return exp_start, exp_end


def get_event_info(ts, periods):
    """Check if timestamp falls within any period."""
    for period in periods:
        if period['start'] <= ts <= period['end']:
            return (period['event'], period['event_name'])
    return ("", "")


def align_fnirs_with_tasks(fnirs_file, periods, output_file, exp_start, exp_end):
    """
    Read fNIRS data, keep only rows inside the experiment window
    [exp_start, exp_end], label them against periods, and save.
    """
    rows_total      = 0   # all rows read from the input file
    rows_kept       = 0   # rows inside the experiment window (saved)
    rows_with_event = 0   # kept rows that matched a labelled period

    with open(fnirs_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader     = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer     = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            rows_total += 1
            ts = parse_iso_timestamp(row['timestamp'].strip())

            if ts is None:
                continue

            # ── trim: skip rows outside the experiment window ──
            if ts < exp_start or ts > exp_end:
                continue

            # ── label the row (existing logic, unchanged) ──
            event, event_name     = get_event_info(ts, periods)
            row['event']          = event      if event      else ""
            row['event_name']     = event_name if event_name else ""

            if event:
                rows_with_event += 1

            writer.writerow(row)
            rows_kept += 1

    rows_trimmed = rows_total - rows_kept
    return rows_total, rows_kept, rows_with_event, rows_trimmed


def process_participant(participant_id, events_file, fnirs_file):
    """Process alignment for one participant. Returns result dict or None on failure."""
    output_file = events_file.parent / f'{participant_id}_fnirs_aligned.csv'

    if not events_file.exists():
        return None, f"events file not found"
    if not fnirs_file.exists():
        return None, f"resampled fNIRS file not found"

    periods = load_task_events(events_file)
    task_count     = sum(1 for p in periods if p['event'] == 'task')
    rest_count     = sum(1 for p in periods if p['event'] == 'rest')
    baseline_count = sum(1 for p in periods if p['event'] == 'baseline')

    # ── compute experiment window ──
    exp_start, exp_end = get_experiment_window(periods)

    rows_total, rows_kept, rows_with_event, rows_trimmed = align_fnirs_with_tasks(
        fnirs_file, periods, output_file, exp_start, exp_end
    )

    labelled_pct = round(100 * rows_with_event / rows_kept, 1) \
                   if rows_kept > 0 else 0.0

    return {
        'subject':      participant_id,
        'samples_in':   rows_total,
        'trimmed':      rows_trimmed,
        'samples_out':  rows_kept,
        'labelled':     rows_with_event,
        'labelled_pct': labelled_pct,
        'task_blocks':  task_count,
        'rest_blocks':  rest_count,
        'baseline':     baseline_count,
    }, None


def print_summary(results):
    print(f"\n{'='*80}")
    print(f"  ALIGNMENT + TRIMMING SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Subject':<12} {'Input':>8} {'Trimmed':>8} {'Kept':>8} "
          f"{'Labelled':>9} {'Label%':>7} {'Tasks':>6} {'Rest':>5}  Status")
    print(f"  {'-'*75}")
    for r in results:
        status = "✅" if r['labelled_pct'] > 80 else "⚠️ low label rate"
        print(f"  {r['subject']:<12} {r['samples_in']:>8} {r['trimmed']:>8} "
              f"{r['samples_out']:>8} {r['labelled']:>9} "
              f"{r['labelled_pct']:>6}% {r['task_blocks']:>6} {r['rest_blocks']:>5}  {status}")
    print(f"{'='*80}\n")


def plot_first_subject():
    """Plot data for the first subject: fNIRS (740, 850nm), IMU, and PPG for first 15s."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import timedelta
    
    base_dir = Path(__file__).parent.parent
    multimodal_dir = base_dir / 'rawdata' / '2nd Trial' / 'Multimodal_data'
    
    # Get first subject folder
    subject_folders = sorted([f for f in multimodal_dir.iterdir() if f.is_dir()])
    if not subject_folders:
        print("❌ No subject folders found")
        return
    
    first_subject_dir = subject_folders[0]
    subject_id = first_subject_dir.name
    
    # Find aligned fNIRS file
    aligned_file = first_subject_dir / f'{subject_id}_fnirs_aligned.csv'
    if not aligned_file.exists():
        print(f"❌ Aligned file not found for {subject_id}")
        return
    
    # Load data
    df = pd.read_csv(aligned_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get first 15 seconds
    start_time = df['timestamp'].iloc[0]
    end_time = start_time + timedelta(seconds=15)
    df_15s = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    # Calculate time relative to start
    df_15s['time_s'] = (df_15s['timestamp'] - start_time).dt.total_seconds()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'{subject_id} - First 15 seconds of Multimodal Data', fontsize=14, fontweight='bold')
    
    # Plot 1: fNIRS (740nm and 850nm) - Channel 0 (PD0)
    ax1 = axes[0]
    ax1.plot(df_15s['time_s'], df_15s['fNIRS (740nm) PD0'], label='740nm (PD0)', linewidth=2, marker='o', markersize=3)
    ax1.plot(df_15s['time_s'], df_15s['fNIRS (850nm) PD0'], label='850nm (PD0)', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax1.set_title('fNIRS: 740nm and 850nm (Channel 0)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IMU data
    ax2 = axes[1]
    ax2.plot(df_15s['time_s'], df_15s['IMU (ax)'], label='X-axis', linewidth=2, marker='o', markersize=3)
    ax2.plot(df_15s['time_s'], df_15s['IMU (ay)'], label='Y-axis', linewidth=2, marker='s', markersize=3)
    ax2.plot(df_15s['time_s'], df_15s['IMU (az)'], label='Z-axis', linewidth=2, marker='^', markersize=3)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Acceleration (g)', fontsize=11)
    ax2.set_title('IMU Data (Accelerometer)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: PPG at 635nm (closest to 640nm)
    ax3 = axes[2]
    ax3.plot(df_15s['time_s'], df_15s['PPG (635nm)'], label='PPG @ 635nm', linewidth=2, marker='o', markersize=3, color='green')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax3.set_title('PPG Data (635nm)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"\n✅ Plotting {subject_id} - First 15 seconds")
    print(f"   Time range: {start_time} to {end_time}")
    print(f"   Samples: {len(df_15s)}")
    plt.show()


def main():
    base_dir        = Path(__file__).parent.parent
    multimodal_dir  = base_dir / 'rawdata' / '2nd Trial' / 'Multimodal_data'

    # Find all subject subfolders
    subject_folders = sorted([f for f in multimodal_dir.iterdir() if f.is_dir()])

    if not subject_folders:
        print(f"❌ No subject folders found in: {multimodal_dir}")
        sys.exit(1)

    # Collect all events files across all subject folders
    events_files = []
    for folder in subject_folders:
        events_files.extend(sorted(folder.glob('*_events.csv')))

    if not events_files:
        print(f"❌ No *_events.csv files found under: {multimodal_dir}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  BATCH fNIRS TASK EVENT ALIGNMENT + TRIMMING")
    print(f"{'='*80}")
    print(f"  Found {len(events_files)} subject(s) to process\n")

    results       = []
    success_count = 0
    fail_count    = 0

    for events_file in events_files:
        # Extract participant ID: first part before underscore
        participant_id = events_file.stem.replace('_events', '').split('_')[0]

        # Find matching resampled fNIRS file in same folder
        fnirs_file = events_file.parent / f'{participant_id}_signals_resampled.csv'

        if not fnirs_file.exists():
            print(f"  ⚠️  {participant_id}: no resampled file found — skipping")
            fail_count += 1
            continue

        result, error = process_participant(participant_id, events_file, fnirs_file)

        if error:
            print(f"  ❌  {participant_id}: {error}")
            fail_count += 1
        else:
            print(f"  ✅  {participant_id}: {result['samples_in']} in → "
                  f"{result['trimmed']} trimmed → {result['samples_out']} kept, "
                  f"{result['labelled_pct']}% labelled")
            results.append(result)
            success_count += 1

    print(f"\n  Done: {success_count} succeeded, {fail_count} failed")

    if results:
        print_summary(results)


if __name__ == '__main__':
    main()
    print("\n" + "="*80)
    print("  PLOTTING FIRST SUBJECT (15 SECONDS)")
    print("="*80)
    plot_first_subject()