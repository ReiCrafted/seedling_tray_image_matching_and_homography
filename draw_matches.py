import cv2
import numpy as np
import os
import json
import shutil

# Configuration
IMAGE_DIR = "images/rgb"
JSON_FILE = "good_matches.json"
OUTPUT_DIR = "matches_visualization"
MIN_INLIERS = 10
MAX_INLIERS = 300

def main():
    # 1. Setup Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 2. Load JSON
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found.")
        return
        
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
        
    # 3. Filter Pairs
    # Filter for > 10 and < 300
    target_pairs = [d for d in data if MIN_INLIERS < d['inliers'] < MAX_INLIERS]
    print(f"Found {len(target_pairs)} pairs to visualize.")
    
    # 4. Initialize Matcher
    orb = cv2.ORB_create(nfeatures=5000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    for i, item in enumerate(target_pairs):
        f1_name = item['source']
        f2_name = item['target']
        inlier_count = item['inliers']
        
        f1_path = os.path.join(IMAGE_DIR, f1_name)
        f2_path = os.path.join(IMAGE_DIR, f2_name)
        
        if not os.path.exists(f1_path) or not os.path.exists(f2_path):
            continue
            
        print(f"Processing {i+1}/{len(target_pairs)}: {f1_name} -> {f2_name} ({inlier_count})")
        
        img1 = cv2.imread(f1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(f2_path, cv2.IMREAD_GRAYSCALE)
        
        # --- STRICT MATCHING LOGIC (COPIED FROM find_matches.py) ---
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            continue
            
        matches12 = matcher.knnMatch(des1, des2, k=2)
        matches21 = matcher.knnMatch(des2, des1, k=2)
        
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
                    
        good_matches = []
        for m1 in good12:
            for m2 in good21:
                if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                    good_matches.append(m1)
                    break
        # -------------------------------------------------------------
        
        # RANSAC to separate inliers for drawing
        if len(good_matches) < 4:
            continue
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
        
        if mask is not None:
             matches_mask = mask.ravel().tolist()
             
             # Draw Matches
             draw_params = dict(matchColor=(0, 255, 0), # Green pairs
                                singlePointColor=None,
                                matchesMask=matches_mask, # Draw only inliers
                                flags=2)
                                
             img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
             
             # Save
             out_name = f"match_{inlier_count}_{f1_name}_to_{f2_name}.jpg"
             cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), img3)

if __name__ == "__main__":
    main()
