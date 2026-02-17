import cv2
import numpy as np
import glob
import os
import re
import json
import sys

# Configuration
IMAGE_DIR = "images/rgb"
OUTPUT_FILE = "good_matches.json"
MIN_INLIERS = 10

def parse_filename(filepath):
    """
    Parses filename to extract coordinates.
    Expected format: x_y_id.png (e.g., 30_600_1757427725.png)
    Returns: (x, y, id, filepath)
    """
    filename = os.path.basename(filepath)
    # Regex to capture integer parts of the filename
    match = re.search(r"(-?\d+)_(-?\d+)_(\d+)", filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        id_val = int(match.group(3))
        return (x, y, id_val, filepath)
    else:
        # Fallback if format is different
        return (0, 0, 0, filepath)

def load_sorted_images():
    pattern = os.path.join(IMAGE_DIR, "*.png")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No images found in {IMAGE_DIR}")
        return []

    # Sort by X, then Y (Vertical overlap check)
    parsed_files = [parse_filename(f) for f in files]
    parsed_files.sort(key=lambda item: (item[0], item[1], item[2]))
    
    sorted_files = [item[3] for item in parsed_files]
    return sorted_files

def main():
    images = load_sorted_images()
    num_images = len(images)
    print(f"Found {num_images} images. Processing pairs...")
    
    good_pairs = []
    
    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=5000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Process sequential pairs
    for i in range(num_images - 1):
        f1 = images[i]
        f2 = images[i+1]
        
        # Simple progress
        if i % 10 == 0:
            print(f"Processing {i}/{num_images-1}...", end='\r')
            
        img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            continue
            
        # Detect and Compute
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            continue
            
        # Match 1 -> 2
        matches12 = matcher.knnMatch(des1, des2, k=2)
        # Match 2 -> 1
        matches21 = matcher.knnMatch(des2, des1, k=2)
        
        # Ratio Test (Stricter: 0.7)
        good12 = []
        for m_n in matches12:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good12.append(m)

        good21 = []
        for m_n in matches21:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good21.append(m)
                    
        # Symmetry Test (Cross Check)
        # Keep matches where queryIdx in 12 matches trainIdx in 21
        good_matches = []
        for m1 in good12:
            for m2 in good21:
                # m1.queryIdx is index in img1 (des1)
                # m1.trainIdx is index in img2 (des2)
                # m2.queryIdx is index in img2 (des2) - this corresponds to m1.trainIdx
                # m2.trainIdx is index in img1 (des1) - this corresponds to m1.queryIdx
                if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                    good_matches.append(m1)
                    break
                    
        if len(good_matches) < MIN_INLIERS:
            continue
            
        # Geometric Verification (RANSAC)
        # Stricter threshold: 1.0 pixel (default was 3.0 or 5.0) for "Perfect" matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
        
        if mask is None:
            inliers = 0
        else:
            inliers = np.sum(mask)
            
        if inliers >= MIN_INLIERS:
            print(f"  FOUND MATCH: {os.path.basename(f1)} -> {os.path.basename(f2)} ({inliers} inliers)")
            good_pairs.append({
                "source": os.path.basename(f1),
                "target": os.path.basename(f2),
                "inliers": int(inliers)
            })
            
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(good_pairs, f, indent=4)
        
    print(f"\nDone. Found {len(good_pairs)} good pairs. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
