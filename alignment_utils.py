import numpy as np
import cv2 as cv

def initialize_detector(method='SIFT'):
    """
    Initializes the feature detector and matcher based on the specified method.
    """
    if method == 'SIFT':
        # Initiate SIFT detector
        detector = cv.SIFT.create()
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
        
        return detector, matcher
    elif method == 'ORB':
        # Initiate ORB detector
        detector = cv.ORB.create(nfeatures=5000)
        # BFMatcher with Hamming distance
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        return detector, matcher
    else:
        raise ValueError(f"Unknown method: {method}")

def detect_and_match(img1, img2, method='SIFT', lowe_ratio=0.7):
    """
    Detects features and matches them between two images.
    Returns keypoints, descriptors, and good matches.
    """
    detector, matcher = initialize_detector(method)
    
    # Find the keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    # Match descriptors
    if method == 'SIFT':
        # SIFT uses KNN matching
         if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
            matches = matcher.knnMatch(des1, des2, k=2)
            # Store all the good matches as per Lowe's ratio test.
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < lowe_ratio * n.distance:
                        good_matches.append(m)
         else:
             good_matches = []
    elif method == 'ORB':
        # ORB/Hamming typically uses KNN matching too for Ratio Test
        if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
            matches = matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < lowe_ratio * n.distance:
                        good_matches.append(m)
        else:
             good_matches = []
    
    return kp1, des1, kp2, des2, good_matches

def compute_homography(kp1, kp2, good_matches, min_match_count=10):
    """
    Computes the Homography matrix if enough matches are found.
    """
    M = None
    mask = None
    dst = None
    
    if len(good_matches) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography transformation matrix
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
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
