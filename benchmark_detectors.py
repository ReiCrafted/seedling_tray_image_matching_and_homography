"""
benchmark_detectors.py

Runs all supported detector/matcher combinations over every image pair in a directory,
measures performance, and writes per-method JSON results + a markdown summary table.

Usage:
    python benchmark_detectors.py --dir images/rgb_27_06_2025 [--max-pairs N]
"""

import cv2 as cv
import numpy as np
import argparse
import json
import os
import glob
import time
import csv
import sys
from pathlib import Path

# Force UTF-8 for console output to avoid cp1252 crash on Windows with emojis
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import alignment_utils as au

# ── All methods to benchmark ───────────────────────────────────────────────────
METHODS = ['SIFT', 'SIFT_BF', 'KAZE', 'AKAZE', 'BRISK', 'ORB']

# ── Helpers ────────────────────────────────────────────────────────────────────

def find_image_pairs(directory):
    """Same pairing logic as the GUI: group by Y, sort by X, pair adjacent X values."""
    files = glob.glob(os.path.join(directory, "*.png")) + \
            glob.glob(os.path.join(directory, "*.jpg"))
    groups = {}
    for f in files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 3:
            try:
                x, y = int(parts[0]), int(parts[1])
                groups.setdefault(y, []).append((x, f))
            except ValueError:
                pass
    pairs = []
    for y in sorted(groups):
        items = sorted(groups[y], key=lambda t: t[0])
        for i in range(len(items) - 1):
            pairs.append((items[i][1], items[i + 1][1]))
    return pairs


def run_pair(img1_path, img2_path, method, orientation='vertical'):
    """
    Run detect → match → homography for one pair + one method.
    Returns a dict of metrics.
    """
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return {"error": "could not load images"}

    h1, w1 = img1.shape[:2]

    # ── Detection + matching ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    kp1, des1, kp2, des2, good_matches = au.detect_and_match(
        img1, img2, method=method, lowe_ratio=0.7, overlap_fraction=0.25, orientation=orientation
    )
    t_detect_match = time.perf_counter() - t0

    n_kp1 = len(kp1) if kp1 else 0
    n_kp2 = len(kp2) if kp2 else 0
    n_good = len(good_matches)

    # ── Homography ─────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    M, mask = au.compute_homography(
        kp1, kp2, good_matches,
        min_match_count=10,
        img_shape=(h1, w1)
    )
    t_homo = time.perf_counter() - t1

    success = M is not None
    inliers = int(mask.sum()) if (mask is not None) else 0
    inlier_ratio = (inliers / n_good) if n_good > 0 else 0.0

    return {
        "img1": os.path.basename(img1_path),
        "img2": os.path.basename(img2_path),
        "kp1_count":       n_kp1,
        "kp2_count":       n_kp2,
        "good_matches":    n_good,
        "inlier_count":    inliers,
        "inlier_ratio":    round(inlier_ratio, 4),
        "time_detect_match_ms": round(t_detect_match * 1000, 1),
        "time_homography_ms":   round(t_homo * 1000, 1),
        "total_time_ms":        round((t_detect_match + t_homo) * 1000, 1),
        "success":         success,
        "M":               M.tolist() if success else None,
    }


def _safe_avg(values):
    return round(sum(values) / len(values), 2) if values else 0.0


def compute_summary_row(method, results):
    """Distil one method's results list into one summary dict."""
    total = len(results)
    if total == 0:
        return {}

    successful   = [r for r in results if r.get("success")]
    failed       = [r for r in results if not r.get("success")]
    n_success    = len(successful)

    good_all      = [r["good_matches"]       for r in results if "good_matches" in r]
    good_ok       = [r["good_matches"]       for r in successful]
    inliers_ok    = [r["inlier_count"]       for r in successful]
    inlier_r_ok   = [r["inlier_ratio"]       for r in successful]
    times_all     = [r["total_time_ms"]      for r in results if "total_time_ms" in r]
    kp1_all       = [r["kp1_count"]          for r in results if "kp1_count" in r]
    kp2_all       = [r["kp2_count"]          for r in results if "kp2_count" in r]

    return {
        "method":               method,
        "total_pairs":          total,
        "success_count":        n_success,
        "fail_count":           total - n_success,
        "success_rate_pct":     round(100 * n_success / total, 1),
        "avg_good_matches_all": _safe_avg(good_all),
        "avg_good_matches_ok":  _safe_avg(good_ok),
        "avg_inliers_ok":       _safe_avg(inliers_ok),
        "avg_inlier_ratio_ok":  _safe_avg(inlier_r_ok),
        "avg_kp1":              _safe_avg(kp1_all),
        "avg_kp2":              _safe_avg(kp2_all),
        "avg_total_time_ms":    _safe_avg(times_all),
        "min_good_matches":     min(good_all) if good_all else 0,
        "max_good_matches":     max(good_all) if good_all else 0,
    }


def write_summary_table(summaries, output_path):
    cols = [
        ("Method",              "method"),
        ("Total Pairs",         "total_pairs"),
        ("✅ Success",          "success_count"),
        ("❌ Failed",           "fail_count"),
        ("Success %",           "success_rate_pct"),
        ("Avg Matches (all)",   "avg_good_matches_all"),
        ("Avg Matches (ok)",    "avg_good_matches_ok"),
        ("Avg Inliers (ok)",    "avg_inliers_ok"),
        ("Avg Inlier Ratio",    "avg_inlier_ratio_ok"),
        ("Avg KP img1",         "avg_kp1"),
        ("Avg KP img2",         "avg_kp2"),
        ("Avg Time/Pair (ms)",  "avg_total_time_ms"),
        ("Min Matches",         "min_good_matches"),
        ("Max Matches",         "max_good_matches"),
    ]

    header   = "| " + " | ".join(h for h, _ in cols) + " |"
    divider  = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for s in summaries:
        row = "| " + " | ".join(str(s.get(k, "")) for _, k in cols) + " |"
        rows.append(row)

    # Sort by success rate descending for easy at-a-glance comparison
    rows_sorted = sorted(rows, key=lambda r: -float(r.split("|")[6].strip() or 0))

    lines = [
        "# Detector / Matcher Benchmark Summary",
        "",
        f"Images directory scanned. Results per method.",
        "",
        header, divider,
        *rows_sorted,
        "",
        "> **Avg Matches (all)**: averaged over ALL pairs, including failures.  ",
        "> **Avg Matches (ok)**: averaged over SUCCESSFUL pairs only.  ",
        r"> **Avg Inlier Ratio**: inliers / good\_matches for successful pairs.  ",
        "> **Avg Time**: total detect+match+RANSAC time per pair in ms.  ",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary table written → {output_path}")


def write_summary_csv(summaries, output_path):
    """Write the same summary data as a CSV file."""
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    # Sort by success_rate_pct descending, same as the markdown table
    rows_sorted = sorted(summaries, key=lambda s: -s.get("success_rate_pct", 0))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"  Summary CSV written    → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark feature detector/matcher combos.")
    parser.add_argument("--dir", default="images/rgb_27_06_2025",
                        help="Directory of images to pair and benchmark.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap the number of pairs (useful for a quick sanity check).")
    parser.add_argument("--out", default="benchmark_results",
                        help="Output folder for result files.")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help=f"Subset of methods to run. All: {METHODS}")
    parser.add_argument("--orientation", choices=['vertical', 'horizontal'], default='vertical',
                        help="Direction of camera sweep overlap (vertical=top/bottom overlap, horizontal=left/right overlap)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print(f"Scanning {args.dir} for image pairs...")
    pairs = find_image_pairs(args.dir)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"Found {len(pairs)} pairs to benchmark across {len(args.methods)} methods.\n")

    summaries = []

    for method in args.methods:
        print(f"═══ {method} {'═' * (20 - len(method))}")
        results = []
        t_method_start = time.perf_counter()

        for i, (p1, p2) in enumerate(pairs):
            r = run_pair(p1, p2, method, orientation=args.orientation)
            results.append(r)

            status = "✅" if r.get("success") else "❌"
            print(f"  [{i+1:>4}/{len(pairs)}] {status}  "
                  f"{r.get('good_matches', 0):>4} matches  "
                  f"{r.get('inlier_count', 0):>4} inliers  "
                  f"{r.get('total_time_ms', 0):>7.1f}ms  "
                  f"{r.get('img1','?')} ↔ {r.get('img2','?')}")

        t_method_total = time.perf_counter() - t_method_start

        # Write per-method JSON
        method_file = out_dir / f"{method}_results.json"
        with open(method_file, "w") as f:
            json.dump({"method": method, "pairs": results}, f, indent=2)
        print(f"  → {method_file}  (total: {t_method_total:.1f}s)\n")

        summaries.append(compute_summary_row(method, results))

    # Write summary markdown + CSV
    summary_md  = out_dir / "summary.md"
    summary_csv = out_dir / "summary.csv"
    write_summary_table(summaries, summary_md)
    write_summary_csv(summaries, summary_csv)

    print("\nAll done.")


if __name__ == "__main__":
    main()
