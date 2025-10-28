import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ─── Configuration ────────────────────────────────────────────────────────────
SEGF_DIR = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/comparisons/AlgoritmoSegFormer/Parameters_mitb1" #mudar para mitb2/mitb3
TRAD_DIR = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/comparisons/AlgoritmoGustavo/parametrospdfsGustavo"
GT_DIR   = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/comparisons/AlgoritmoSegFormer/manualground-truth_Parameters"

NUM_VIDEOS = 23
EPS = 1e-8


def load_csv(path, label):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {label} file not found: {path}")
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        sys.exit(f"ERROR loading {label} from {path}: {e}")
    return df


def compute_summary(df_seg, df_trad, df_gt):
    metrics = df_gt.index.intersection(df_seg.index).intersection(df_trad.index)
    frames  = df_gt.columns.intersection(df_seg.columns).intersection(df_trad.columns)
    if metrics.empty or frames.empty:
        raise ValueError("No common metrics or frames to compare.")

    segc = df_seg.loc[metrics, frames]
    tradc = df_trad.loc[metrics, frames]
    gtc  = df_gt.loc[metrics, frames]

    flat_seg  = segc.values.flatten()
    flat_trad = tradc.values.flatten()
    flat_gt   = gtc.values.flatten()

    seg_better = trad_better = ties = 0
    for s, g, t in zip(flat_seg, flat_trad, flat_gt):
        err_s = abs(s - t)
        err_g = abs(g - t)
        if err_s + EPS < err_g:
            seg_better += 1
        elif err_g + EPS < err_s:
            trad_better += 1
        else:
            ties += 1

    non_tied = seg_better + trad_better
    acc_seg  = (seg_better / non_tied * 100) if non_tied > 0 else np.nan
    acc_trad = (trad_better / non_tied * 100) if non_tied > 0 else np.nan

    def global_stats(pred, gt):
        diff = np.abs(pred - gt)
        mae  = diff.mean()
        rmse = np.sqrt((diff**2).mean())
        try:
            r, _ = pearsonr(pred, gt)
        except Exception:
            r = np.nan
        return mae, rmse, r

    mae_s, rmse_s, r_s       = global_stats(flat_seg, flat_gt)
    mae_t, rmse_t, r_t       = global_stats(flat_trad, flat_gt)

    return {
        'seg_cells_better': seg_better,
        'trad_cells_better': trad_better,
        'equal_error_cells': ties,
        'seg_accuracy_pct': acc_seg,
        'trad_accuracy_pct': acc_trad,
        'mae_seg': mae_s,
        'rmse_seg': rmse_s,
        'r_seg': r_s,
        'mae_trad': mae_t,
        'rmse_trad': rmse_t,
        'r_trad': r_t,
    }


def main():
    results = []
    for i in range(1, NUM_VIDEOS + 1):
        seg_path  = os.path.join(SEGF_DIR,  f"video{i}_mitb1_chordpoints.csv") #mudar para "video{i}_mitb2_chordpoints.csv"/"video{i}_mitb3_chordpoints.csv"
        trad_path = os.path.join(TRAD_DIR,  f"video{i}_parameters.csv")
        gt_path   = os.path.join(GT_DIR,    f"video{i}_chordpoints_manual.csv")

        df_seg  = load_csv(seg_path,  "SegFormer")
        df_trad = load_csv(trad_path, "Traditional")
        df_gt   = load_csv(gt_path,   "GroundTruth")

        summary = compute_summary(df_seg, df_trad, df_gt)
        summary['video'] = f"video{i}"

        if summary['seg_cells_better'] > summary['trad_cells_better']:
            summary['preferred_approach'] = 'SegFormer'
        elif summary['trad_cells_better'] > summary['seg_cells_better']:
            summary['preferred_approach'] = 'Traditional'
        else:
            summary['preferred_approach'] = 'Tie'

        results.append(summary)

    df_res = pd.DataFrame(results)

    cols = [
        'video',
        'seg_accuracy_pct', 'trad_accuracy_pct',
        'mae_seg', 'rmse_seg', 'r_seg',
        'mae_trad', 'rmse_trad', 'r_trad',
        'preferred_approach'
    ]
    df_res = df_res[cols]

    float_cols = [c for c in df_res.columns if c not in ['video', 'preferred_approach']]
    df_res[float_cols] = df_res[float_cols].astype(float).round(2)

    # Compact per-video table
    print("\n=== Per-video comparison summary ===")
    print(df_res.to_string(index=False))

    # Aggregate summary
    total_seg_better  = int(sum(r['seg_cells_better'] for r in results))
    total_trad_better = int(sum(r['trad_cells_better'] for r in results))
    total_equal       = int(sum(r['equal_error_cells'] for r in results))

    if total_seg_better > total_trad_better:
        overall = 'SegFormer'
    elif total_trad_better > total_seg_better:
        overall = 'Traditional'
    else:
        overall = 'Tie'

    print("\n==== Overall summary ====")
    print(f"SegFormer better:      {total_seg_better}")
    print(f"Traditional better:    {total_trad_better}")
    print(f"Ties:                  {total_equal}")
    print(f"Overall Best Approach: {overall}")

    # Save CSV
    out_csv = "mitb1_comparison_summary_all_videos.csv" #mudar para mitb2/mitb3
    df_res.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"\nSaved summary CSV to {out_csv}")


if __name__ == "__main__":
    main()
