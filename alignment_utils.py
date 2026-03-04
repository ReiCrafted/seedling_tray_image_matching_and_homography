import numpy as np
import cv2 as cv
import os
import json

def initialize_detector(method='SIFT'):
    """
    Initializes a feature detector and a corresponding matcher.
    Supported methods: SIFT, SIFT_BF, ORB, AKAZE, BRISK, KAZE
    """
    FLANN_INDEX_KDTREE = 1
    flann_float = cv.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )
    # FLANN for binary LSH descriptors
    FLANN_INDEX_LSH = 6
    flann_binary = cv.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2),
        dict(checks=50)
    )
    bf_hamming = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    bf_l2      = cv.BFMatcher(cv.NORM_L2,      crossCheck=False)

    methods = {
        'SIFT':    (cv.SIFT.create(),                                     flann_float),
        'SIFT_BF': (cv.SIFT.create(),                                     bf_l2),
        'ORB':     (cv.ORB.create(nfeatures=5000),                        bf_hamming),
        'AKAZE':   (cv.AKAZE.create(),                                    bf_hamming),
        'BRISK':   (cv.BRISK.create(),                                    bf_hamming),
        'KAZE':    (cv.KAZE.create(),                                     flann_float),
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods)}")
    return methods[method]

# Methods that produce binary (Hamming) descriptors — use ratio-test matching directly
_BINARY_METHODS = {'ORB', 'AKAZE', 'BRISK'}
# Methods with float descriptors — FLANN or BF L2
_FLOAT_METHODS  = {'SIFT', 'SIFT_BF', 'KAZE'}

def detect_and_match(img1, img2, method='SIFT', lowe_ratio=0.7, overlap_fraction=0.20, orientation='vertical'):
    """
    Detects features and matches them between two images.
    Only searches within the expected overlap band:
      - vertical: bottom 20% of img1, top 20% of img2
      - horizontal: right 20% of img1, left 20% of img2
    Keypoint coordinates are remapped to full-image space before returning.
    Returns keypoints, descriptors, and good matches.
    """
    detector, matcher = initialize_detector(method)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    crop1_x0, crop1_y0 = 0, 0
    crop2_x0, crop2_y0 = 0, 0

    if orientation == 'vertical':
        # img1 TOP overlaps img2 BOTTOM (Y-axis sweep)
        crop1 = img1[0 : int(h1 * overlap_fraction), :]
        crop2_y0 = int(h2 * (1.0 - overlap_fraction))
        crop2 = img2[crop2_y0 : h2, :]
    elif orientation == 'horizontal':
        # img1 RIGHT overlaps img2 LEFT (X-axis sweep, 90deg rotated images)
        crop1_x0 = int(w1 * (1.0 - overlap_fraction))
        crop1 = img1[:, crop1_x0 : w1]
        crop2 = img2[:, 0 : int(w2 * overlap_fraction)]
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # Find keypoints in crop regions
    kp1_crop, des1 = detector.detectAndCompute(crop1, None)
    kp2_crop, des2 = detector.detectAndCompute(crop2, None)

    # Remap keypoint coordinates back to full-image space
    def remap(kps, x_offset, y_offset):
        return [cv.KeyPoint(kp.pt[0] + x_offset, kp.pt[1] + y_offset, kp.size,
                            kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in kps]

    kp1 = remap(kp1_crop, crop1_x0, crop1_y0)
    kp2 = remap(kp2_crop, crop2_x0, crop2_y0)

    # Match descriptors with Lowe's ratio test
    good_matches = []
    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        matches = matcher.knnMatch(des1, des2, k=2)
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < lowe_ratio * n.distance:
                    good_matches.append(m)

    return kp1, des1, kp2, des2, good_matches


def is_valid_homography(M, h, w):
    if M is None:
        return False
        
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M).reshape(-1, 2)
    
    # 1. Check for convexity (no crossed edges)
    if not cv.isContourConvex(np.int32(dst)):
        print("Homography rejected: Resulting polygon is not convex.")
        return False
        
    # 2. Check scale change (assuming camera height is constant, scale shouldn't change much)
    # Area of original image vs area of projected image
    orig_area = w * h
    proj_area = cv.contourArea(np.float32(dst))
    
    if proj_area == 0:
        return False
        
    scale_ratio = proj_area / orig_area
    # Allow 10% variation in area to accommodate mild camera sway/tilt 
    # (prev. 40% was needed before undistortion, but since we mathematically un-distort first, 10% is correct)
    if scale_ratio < 0.9 or scale_ratio > 1.1:
        print(f"Homography rejected: Unreasonable scale change (ratio: {scale_ratio:.2f}).")
        return False
        
    # 3. Check for extreme perspective distortion
    # The determinant of the top-left 2x2 matrix gives a rough idea of the affine scale
    det = np.linalg.det(M[0:2, 0:2])
    if det < 0.9 or det > 1.1:
        print(f"Homography rejected: Extreme affine scaling detected (det: {det:.2f}).")
        return False

    return True

def compute_homography(kp1, kp2, good_matches, min_match_count=10, img_shape=(1080, 1920)):
    """
    Computes the Homography matrix if enough matches are found and validates the result.
    Pass img_shape (h, w) to validate the homography properly.
    """
    M = None
    mask = None
    
    if len(good_matches) >= min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Apply lens undistortion to the geometric points mathematically if calibration exists
        if os.path.exists('camera_params.json'):
            try:
                with open('camera_params.json', 'r') as f:
                    params = json.load(f)
                K = np.array(params['camera_matrix'])
                dist = np.array(params['dist_coeffs'])
                src_pts = cv.undistortPoints(src_pts, K, dist, P=K)
                dst_pts = cv.undistortPoints(dst_pts, K, dist, P=K)
            except Exception as e:
                print(f"Warning: Failed to load camera_params.json: {e}")

        # Increased RANSAC threshold to 15.0 for more robust matching
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 15.0)
        
        if M is not None:
            # Validate the homography
            h, w = img_shape
            if not is_valid_homography(M, h, w):
                return None, None
                
    return M, mask

def calculate_projection_and_overlap(M, img1_shape, img2_shape):
    """
    Calculates the projected area and overlap metrics.
    """
    h, w = img1_shape
    h2, w2 = img2_shape
    
    # Points of the first image
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    # Project points to second image
    dst = cv.perspectiveTransform(pts, M)
    
    # Projected polygon area in img2 (in pixels^2)
    poly_proj = dst.reshape(-1, 2).astype(np.float32)
    area_proj_px = float(abs(cv.contourArea(poly_proj)))

    # Intersection with the image bounds (to measure on-canvas overlap area)
    img_rect = np.array([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]], dtype=np.float32)

    # intersectConvexConvex expects convex polygons in consistent order
    try:
        inter_area_px, _ = cv.intersectConvexConvex(poly_proj, img_rect)
    except Exception:
        # Fallback if intersection fails (e.g., self-intersection)
        inter_area_px = 0.0

    # Relative metrics
    img2_area_px = float(h2 * w2)
    rel_to_img2 = (inter_area_px / img2_area_px) if img2_area_px > 0 else 0.0
    rel_on_canvas_of_proj = (inter_area_px / area_proj_px) if area_proj_px > 0 else 0.0
    
    return {
        "area_proj_px": area_proj_px,
        "inter_area_px": inter_area_px,
        "rel_to_img2": rel_to_img2,
        "rel_on_canvas_of_proj": rel_on_canvas_of_proj,
        "dst": dst
    }

def visualize_results(img1_color, kp1, img2_color, kp2, good_matches, mask, dst, overlay_lines):
    """
    Visualizes the matches and the overlapping area.
    """
    # Draw the projected quadrilateral and stats on the color image
    img_result = img2_color.copy()
    if dst is not None:
        img_result = cv.polylines(img_result, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)

        y0 = 30
        for i, line in enumerate(overlay_lines):
            y = y0 + i * 28
            cv.putText(img_result, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

    # Draw inlier matches
    matchesMask = mask.ravel().tolist() if mask is not None else None
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    img_matches = cv.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None, **draw_params)
    
    return img_result, img_matches
