import cv2
import numpy as np
import glob
import os
import json
import sys
import re

# Configuration
IMAGE_DIR = os.path.join("images", "rgb")
OUTPUT_FILE = "ground_truth.json"
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
CANVAS_PADDING = 0.5 # 50% padding around the image

# Global state
current_pair_index = 0
image_pairs = []
results = []
current_homography = None
drag_point_idx = -1
base_corners_pts = None # To store current corner points of the overlay (in DISPLAY coordinates, relative to CANVAS)
display_scale = 1.0

def parse_filename(filepath):
    """
    Parses '30_600_1757427725.png' -> (x=30, y=600, id=1757427725)
    Returns tuple (x, y, id, filepath)
    """
    basename = os.path.basename(filepath)
    # Regex to find integer segments
    parts = re.findall(r'\d+', basename)
    if len(parts) >= 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]), filepath)
    else:
        # Fallback
        return (0, 0, 0, filepath)

def load_images():
    """Loads images from directory, sorts them by Coordinate (Y then X), and creates pairs."""
    global image_pairs
    
    pattern = os.path.join(IMAGE_DIR, "*.png")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No images found in {IMAGE_DIR}")
        return []

    # Parse and Sort
    # Sorting by X first, then Y (Column-wise order) to test Vertical Overlap
    parsed_files = [parse_filename(f) for f in files]
    # Sort key: (x, y, id)
    parsed_files.sort(key=lambda item: (item[0], item[1], item[2]))
    
    sorted_files = [item[3] for item in parsed_files]

    # Create sequential pairs
    for i in range(len(sorted_files) - 1):
        image_pairs.append((sorted_files[i], sorted_files[i+1]))
        
    print(f"Found {len(sorted_files)} images, created {len(image_pairs)} pairs.")
    # Debug print first few to verify sort
    for i in range(min(5, len(sorted_files))):
        print(f"Sorted {i}: {os.path.basename(sorted_files[i])}")
        
    return image_pairs

def get_initial_homography(img1, img2, f1_path, f2_path):
    """
    Computes initial homography using high-accuracy SIFT and potentially MAGSAC.
    Restricts matching to the expected overlap region based on filename coordinates.
    Operates on full resolution images.
    """
    # 1. Determine ROI
    x1, y1, _, _ = parse_filename(f1_path)
    x2, y2, _, _ = parse_filename(f2_path)
    
    h, w = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Overlap assumption (25% to be safe for 20% overlap)
    OVERLAP_PCT = 0.25
    
    # Define ROI slices + offsets for Keypoint adjustment
    # Format: (slice_y, slice_x, offset_x, offset_y)
    roi1 = (slice(0, h), slice(0, w), 0, 0)
    roi2 = (slice(0, h2), slice(0, w2), 0, 0)
    
    dx = x2 - x1
    dy = y2 - y1
    
    direction = "Unknown"
    
    # Determine dominant direction
    # Note: Filenames like 30_600... x=30 (mm), y=600 (mm).
    # Assuming positive X is Right, positive Y is Down (or Up, doesn't matter for adjacency relative check).
    
    if abs(dx) > abs(dy): # Horizontal Neighbor
        if dx > 0: # Image 2 is to the RIGHT of Image 1
            direction = "Right"
            # Img1: Right Edge matches Img2: Left Edge
            roi1 = (slice(0, h), slice(int(w * (1 - OVERLAP_PCT)), w), int(w * (1 - OVERLAP_PCT)), 0)
            roi2 = (slice(0, h2), slice(0, int(w2 * OVERLAP_PCT)), 0, 0)
        else: # Image 2 is to the LEFT of Image 1
            direction = "Left"
            # Img1: Left Edge matches Img2: Right Edge
            roi1 = (slice(0, h), slice(0, int(w * OVERLAP_PCT)), 0, 0)
            roi2 = (slice(0, h2), slice(int(w2 * (1 - OVERLAP_PCT)), w2), int(w2 * (1 - OVERLAP_PCT)), 0)
            
    elif abs(dy) >= abs(dx) and abs(dy) > 0: # Vertical Neighbor
        if dy > 0: # Image 2 is BELOW Image 1 (assuming Y increases downwards or just distinct)
             # "Below" in raster usually means next row.
             # Img1: Bottom Edge matches Img2: Top Edge
             direction = "Vertical (Next Row?)"
             roi1 = (slice(int(h * (1 - OVERLAP_PCT)), h), slice(0, w), 0, int(h * (1 - OVERLAP_PCT)))
             roi2 = (slice(0, int(h2 * OVERLAP_PCT)), slice(0, w2), 0, 0)
        else: # Image 2 is ABOVE Image 1
             direction = "Vertical (Prev Row?)"
             roi1 = (slice(0, int(h * OVERLAP_PCT)), slice(0, w), 0, 0)
             roi2 = (slice(int(h2 * (1 - OVERLAP_PCT)), h2), slice(0, w2), 0, int(h2 * (1 - OVERLAP_PCT)))
             
    print(f"  Direction: {direction} (dx={dx}, dy={dy}). Constraining ORB to 25% ROI.")

    # Crop Images
    img1_roi = img1[roi1[0], roi1[1]]
    img2_roi = img2[roi2[0], roi2[1]]
    
    # Initialize ORB (User requested ORB instead of SIFT)
    orb = cv2.ORB_create(nfeatures=10000)
    
    kp1, des1 = orb.detectAndCompute(img1_roi, None)
    kp2, des2 = orb.detectAndCompute(img2_roi, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.eye(3, dtype=np.float32), 0

    # Offset Keypoints back to global coordinates
    for k in kp1:
        k.pt = (k.pt[0] + roi1[2], k.pt[1] + roi1[3])
    for k in kp2:
        k.pt = (k.pt[0] + roi2[2], k.pt[1] + roi2[3])

    # BFMatcher with Hamming distance for ORB
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    matches = matcher.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
    match_count = len(good_matches)
    print(f"  Matches found in ROI (ORB): {match_count}")

    if match_count < 4:
        print("  ! Not enough matches for homography.")
        return np.eye(3, dtype=np.float32), match_count
        
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # User requested reverting to Homography for perspective handling
    # Try using MAGSAC++ if available, else RANSAC with strict threshold
    method = cv2.RANSAC
    if hasattr(cv2, 'USAC_MAGSAC'):
        method = cv2.USAC_MAGSAC
        
    M, mask = cv2.findHomography(src_pts, dst_pts, method, 3.0)
    
    valid_homography = False
    if M is not None:
        # --- Sanity Checks ---
        # 1. Project corners to check for convexity and twisting
        h, w = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        src_corners_check = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        dst_corners_check = cv2.perspectiveTransform(src_corners_check, M)
        
        # Check Convexity
        is_convex = cv2.isContourConvex(dst_corners_check.astype(np.int32))
        
        # Check Area (prevent collapsing or explosions)
        area = cv2.contourArea(dst_corners_check)
        img2_area = w2 * h2
        is_area_ok = (area > img2_area * 0.1) and (area < img2_area * 10.0)
        
        if is_convex and is_area_ok:
            valid_homography = True
        else:
             print(f"  ! Homography rejected (Convex: {is_convex}, Area: {is_area_ok}). Falling back to Affine.")

    if valid_homography:
        return M, match_count
        
    # Fallback to Affine Partial (Rotation + Translation + Scale)
    print("  Attempting Affine Fallback...")
    M_affine, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if M_affine is None:
        return np.eye(3, dtype=np.float32), match_count
        
    # Convert 2x3 Affine to 3x3 Homography
    M = np.eye(3, dtype=np.float32)
    M[:2, :] = M_affine
    
    return M, match_count

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    """
    Brightness = -255 to 255
    Contrast = -127 to 127
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        
    return buf

def get_display_scale(h, w):
    # We want to fit the "Canvas" which is roughly 2x image size
    target_canvas_w = w * (1 + 2*CANVAS_PADDING)
    target_canvas_h = h * (1 + 2*CANVAS_PADDING)
    
    # Fit into 720p/1080p ish window (accounting for canvas size)
    # The canvas is roughly 2.6x the image size in each dim (if padding is 0.8)
    # Let's target a max window size of 1600x900
    
    max_win_w = 1600
    max_win_h = 900
    
    scale_w = max_win_w / target_canvas_w
    scale_h = max_win_h / target_canvas_h
    return min(scale_w, scale_h, 1.0) # Don't upscale

def update_display(win_name, canvas_base, img2_warped, corners, offset_x, offset_y):
    """
    Composites the images on the canvas and draws control points.
    canvas_base: The large black canvas with img1 already placed in center.
    img2_warped: Overlay image warped to perspective (same size as canvas).
    corners: Control points (corners of overlay relative to canvas 0,0)
    """
    
    # Composite
    # White background logic:
    # canvas_base is White (255) outside img1.
    # img2_warped is White (255) outside img2 (if warped with borderValue=255).
    # addWeighted(0.35, 0.65)
    # Outside: 0.35*255 + 0.65*255 = 255 (White).
    # Overlap: 0.35*img1 + 0.65*img2.
    
    display = cv2.addWeighted(canvas_base, 0.35, img2_warped, 0.65, 0)
    
    # Draw corners and lines
    pts = corners.astype(np.int32)
    cv2.polylines(display, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
    
    for i, pt in enumerate(pts):
        cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
        # Helper text for corner ID
        cv2.putText(display, str(i), (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Draw a rectangle around the base image to show where it is "anchored"
    # Base image is at (offset_x, offset_y)
    h, w, _ = canvas_base.shape # Actually we need original img1 size here... 
    # But we can infer it or pass it. 
    # Let's just rely on the visual image content for now.

    cv2.putText(display, "SPACE: Save/Next | ESC: Exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(win_name, display)

def mouse_callback(event, x, y, flags, param):
    global drag_point_idx, base_corners_pts, current_homography
    
    img2_h, img2_w = param['img2_display_size']
    offset_x, offset_y = param['offset']
    
    # Source corners of the overlay image (0,0 is top-left of the overlay image itself)
    src_corners = np.float32([[0, 0], [img2_w, 0], [img2_w, img2_h], [0, img2_h]])
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking near a corner
        min_dist = 20.0
        idx = -1
        for i, pt in enumerate(base_corners_pts):
            dist = np.linalg.norm(pt - np.array([x, y]))
            if dist < min_dist:
                min_dist = dist
                idx = i
        drag_point_idx = idx

    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_point_idx != -1:
            base_corners_pts[drag_point_idx] = [x, y]
            
            # Recompute Homography. 
            # Warping logic: Source (0,0) -> Dest (Canvas Coord).
            # To preserve logic relative to Base Image (at offset_x, offset_y):
            # The Homography we actually care about for saving is (Src -> Base Image 0,0).
            # But here we are calculating (Src -> Canvas Coord).
            # Canvas Coord = Base Image Coord + Offset.
            # So, H_canvas maps to (Base + Offset).
            
            M_canvas = cv2.getPerspectiveTransform(src_corners, base_corners_pts)
            
            # Re-warp and update display
            # We warp into the full canvas size
            canvas_h, canvas_w = param['canvas_shape'][:2]
            img2_warped = cv2.warpPerspective(param['img2'], M_canvas, (canvas_w, canvas_h), borderValue=(255, 255, 255))
            
            update_display(param['win_name'], param['canvas_base'], img2_warped, base_corners_pts, offset_x, offset_y)

    elif event == cv2.EVENT_LBUTTONUP:
        drag_point_idx = -1

def save_current_result(pair, corners_display, offset):
    global display_scale
    
    # corners_display are in Canvas Coordinates.
    # We need them in Base Image Coordinates (Original Resolution).
    
    offset_x, offset_y = offset
    
    # 1. Remove Canvas Offset
    corners_relative_disp = corners_display - np.array([offset_x, offset_y])
    
    # 2. Scale back to Full Resolution
    corners_full = corners_relative_disp / display_scale
    
    entry = {
        "pair_index": current_pair_index,
        "base_filename": os.path.basename(pair[0]),
        "overlay_filename": os.path.basename(pair[1]),
        "corners": corners_full.tolist()
    }
    
    # Check if entry already exists (update it)
    found = False
    for i, res in enumerate(results):
        if res.get("base_filename") == entry["base_filename"] and \
           res.get("overlay_filename") == entry["overlay_filename"]:
            results[i] = entry
            found = True
            break
    if not found:
        results.append(entry)
        
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved pair {current_pair_index}")

def main():
    global current_pair_index, image_pairs, current_homography, base_corners_pts, results, display_scale
    
    # Load existing results if any
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results.")
        except:
            print("Could not load existing results, starting fresh.")
            results = [] # Always start fresh

    load_images()
    
    if not image_pairs:
        return

    # Jump to middle index for testing
    current_pair_index = 50
    if current_pair_index >= len(image_pairs):
        current_pair_index = 0

    win_name = "Ground Truth Generator"
    # Allow window resize
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
    
    while current_pair_index < len(image_pairs):
        f1, f2 = image_pairs[current_pair_index]
        print(f"Processing Pair {current_pair_index}: {os.path.basename(f1)} -> {os.path.basename(f2)}")
        
        # Load Full Resolution
        img1_full = cv2.imread(f1)
        img2_full = cv2.imread(f2)
        
        if img1_full is None or img2_full is None:
            print("Error loading images, skipping.")
            current_pair_index += 1
            continue

        h_full, w_full = img1_full.shape[:2]
        h2_full, w2_full = img2_full.shape[:2]
        
        # Calculate Scale
        display_scale = get_display_scale(h_full, w_full)
        
        # Create Display Images (Scaled)
        new_w = int(w_full * display_scale)
        new_h = int(h_full * display_scale)
        img1_disp = cv2.resize(img1_full, (new_w, new_h)) 
        img2_disp = cv2.resize(img2_full, (new_w, new_h))
        
        # Apply Brightness/Contrast Boost for Display
        # Using fixed values for now based on user feedback (dark images)
        # Brightness ~60 means we need significant boost e.g. +50 brightness, +30 contrast
        img1_disp = apply_brightness_contrast(img1_disp, brightness=50, contrast=30)
        img2_disp = apply_brightness_contrast(img2_disp, brightness=50, contrast=30)
        
        h2_disp = int(h2_full * display_scale)
        w2_disp = int(w2_full * display_scale)

        # Create centered Canvas
        # Canvas should be large enough to hold the base image plus lots of padding
        canvas_w = int(new_w * (1 + 2*CANVAS_PADDING))
        canvas_h = int(new_h * (1 + 2*CANVAS_PADDING))
        
        # White background (255)
        canvas_base = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        
        # Paste img1 in center
        offset_x = int(new_w * CANVAS_PADDING)
        offset_y = int(new_h * CANVAS_PADDING)
        
        canvas_base[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img1_disp

        # Check for existing result
        existing_corners_full = None
        for res in results:
            if res["base_filename"] == os.path.basename(f1) and \
               res["overlay_filename"] == os.path.basename(f2):
                existing_corners_full = np.array(res["corners"], dtype=np.float32)
                break
        
        match_count = -1
        
        if existing_corners_full is not None:
            print("  Loaded existing manual adjustment.")
            # Convert Full Res Corners (relative to Base Image) -> Display Canvas Corners
            base_corners_pts = ((existing_corners_full * display_scale) + np.array([offset_x, offset_y])).astype(np.float32)
        else:
            # Auto Estimate on Full Res
            print("  Auto-estimating overlap (Full Res with ROI)...")
            M_full, match_count = get_initial_homography(img1_full, img2_full, f1, f2)
            
            # Project corners using Full Res M to get coords relative to Base Image Top-Left (Full Res)
            src_corners_full = np.float32([[0, 0], [w2_full, 0], [w2_full, h2_full], [0, h2_full]]).reshape(-1, 1, 2)
            dst_corners_full = cv2.perspectiveTransform(src_corners_full, M_full)
            dst_corners_full = dst_corners_full.reshape(4, 2)
            
            # Convert to Display Canvas Coords
            base_corners_pts = ((dst_corners_full * display_scale) + np.array([offset_x, offset_y])).astype(np.float32)

        # Current Homography on Canvas (Src Display -> Canvas Points)
        src_corners_disp = np.float32([[0, 0], [w2_disp, 0], [w2_disp, h2_disp], [0, h2_disp]])
        M_canvas = cv2.getPerspectiveTransform(src_corners_disp, base_corners_pts)

        # Draw initial warped overlay on canvas
        # Use white border for warping so it blends into white background
        img2_warped = cv2.warpPerspective(img2_disp, M_canvas, (canvas_w, canvas_h), borderValue=(255, 255, 255))
        update_display(win_name, canvas_base, img2_warped, base_corners_pts, offset_x, offset_y)
        
        # Mouse Callback
        cv2.setMouseCallback(win_name, mouse_callback, {
            'img2': img2_disp, 
            'img2_display_size': (h2_disp, w2_disp), 
            'target_display_size': (2,2), # Unused?
            'win_name': win_name,
            'canvas_base': canvas_base,
            'canvas_shape': (canvas_h, canvas_w),
            'offset': (offset_x, offset_y)
        })
        
        print(f"Controls: [Space] Save & Next, [Esc] Exit. Processing {current_pair_index+1}/{len(image_pairs)}")
        if match_count != -1 and match_count < 4:
             print("  WARNING: Low match count! Auto-alignment may be random.")
        
        # Loop for interaction
        next_pair = False
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == 27: # Esc
                sys.exit(0)
            elif key == 32: # Space
                save_current_result((f1, f2), base_corners_pts, (offset_x, offset_y))
                next_pair = True
                break
            elif key == ord('r'):
                 pass 
        
        if next_pair:
            current_pair_index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
