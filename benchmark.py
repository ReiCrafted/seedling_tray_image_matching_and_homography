import time
import os
import glob
import cv2 as cv
import pandas as pd
import alignment_utils as au

# Configuration
IMAGE_DIR = "images"
RESULTS_FILE = "benchmark_results.csv"

def run_benchmark(image_dir, method='SIFT'):
    """
    Runs the benchmark on all pairs of images in the directory.
    Assumes images are named in a way that pairs can be identified or simply runs on all sequential pairs.
    For this implementation, we will look for pairs or just take the first two found as a demo.
    """
    
    # Gather images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    if len(image_paths) < 2:
        print("Not enough images to benchmark.")
        return

    results = []

    # Simple sequential pair benchmarking (0-1, 1-2, 2-3...)
    # In a real grid, you might have specific wiring logic.
    for i in range(len(image_paths) - 1):
        path1 = image_paths[i]
        path2 = image_paths[i+1]
        
        pair_id = f"{os.path.basename(path1)}-{os.path.basename(path2)}"
        
        print(f"Benchmarking {pair_id} with {method}...")

        try:
            img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                print(f"Error reading images for pair {pair_id}")
                continue

            start_time = time.time()
            
            # 1. Detect and Match
            kp1, des1, kp2, des2, good_matches = au.detect_and_match(img1, img2, method=method)
            
            # 2. Homography
            M, mask = au.compute_homography(kp1, kp2, good_matches)
            
            # 3. Metrics
            overlap_pct = 0.0
            if M is not None:
                metrics = au.calculate_projection_and_overlap(M, img1.shape, img2.shape)
                overlap_pct = metrics['rel_to_img2'] * 100
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            results.append({
                "Algorithm": method,
                "Pair": pair_id,
                "Execution Time (ms)": duration_ms,
                "Inliers": len(good_matches),
                "Overlap %": overlap_pct,
                "Success": M is not None
            })
            
        except Exception as e:
            print(f"Failed on {pair_id}: {e}")
            results.append({
                "Algorithm": method,
                "Pair": pair_id,
                "Execution Time (ms)": 0,
                "Inliers": 0,
                "Overlap %": 0,
                "Success": False
            })

    return results

if __name__ == "__main__":
    
    all_results = []
    
    # Run SIFT
    print("--- Running SIFT Benchmark ---")
    sift_results = run_benchmark(IMAGE_DIR, method='SIFT')
    if sift_results:
        all_results.extend(sift_results)
        
    # Run ORB (Future proofing, as requested in plan)
    print("\n--- Running ORB Benchmark ---")
    orb_results = run_benchmark(IMAGE_DIR, method='ORB')
    if orb_results:
        all_results.extend(orb_results)
        
    # Save Results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\nResults saved to {RESULTS_FILE}")
        print(df)
    else:
        print("No results generated.")
