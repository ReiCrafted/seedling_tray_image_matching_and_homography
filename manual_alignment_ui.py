import cv2 as cv
import numpy as np
import argparse
import json
import alignment_utils as au
import os
import ctypes
import glob
import time

# Enable OpenCL if available for transparent GPU acceleration via UMat
try:
    cv.ocl.setUseOpenCL(True)
    print("OpenCL acceleration enabled:", cv.ocl.haveOpenCL())
except Exception as e:
    print("Could not initialize OpenCL:", e)

def find_image_pairs(directory):
    """Scans a directory for images and groups them by Y coordinate, pairing adjacent X coordinates."""
    files = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
    groups = {}
    for f in files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 3:
            try:
                x = int(parts[0])
                y = int(parts[1])
                if y not in groups:
                    groups[y] = []
                groups[y].append((x, f))
            except ValueError:
                pass
                
    pairs = []
    # Sort y to be deterministic
    for y in sorted(groups.keys()):
        groups[y].sort(key=lambda item: item[0])
        items = groups[y]
        for i in range(len(items) - 1):
            # smaller X is img1 (moving), bigger X is img2 (static canvas)
            pairs.append((items[i][1], items[i+1][1]))
            
    return pairs

class ManualAlignmentUI:
    def __init__(self, pairs, start_idx=0):
        self.pairs = pairs
        self.current_idx = start_idx
        self.saved_matrices = {} # Stores custom matrices
        self.homography_status = {} # Tracks whether initial homography succeeded or failed
        
        if len(self.pairs) == 0:
            raise ValueError("No image pairs provided.")
            
        self.window_name = "Manual Alignment UI"
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        
        self.canvas_w = 1280
        self.canvas_h = 720
        cv.resizeWindow(self.window_name, self.canvas_w, self.canvas_h)
        cv.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.dragging_idx = -1
        self.panning = False
        self.mode = 0  # 0: Alpha, 1: Difference, 2: Anaglyph, 3: Edges
        self.modes_count = 4
        self.help_visible = True
        
        # State variables
        self.img1_path = None
        self.img2_path = None
        self.img1 = None
        self.img2 = None
        # UMat versions for faster processing
        self.u_img1 = None 
        self.u_img2 = None
        
        self.h1 = 0; self.w1 = 0; self.h2 = 0; self.w2 = 0
        self.src_pts = None
        self.dst_pts = None
        self.M = None
        self.zoom = 1.0
        self.offset_x = 0.0; self.offset_y = 0.0
        self.is_valid_auto_match = False
        self.save_status = None  # None | "saving" | "saved"
        self.save_status_time = 0.0
        self.needs_render = True
        
        self.load_pair()
        
    def load_pair(self):
        self.img1_path, self.img2_path = self.pairs[self.current_idx]
        print(f"\n--- Loading Pair {self.current_idx + 1}/{len(self.pairs)} ---")
        print(f"Img1 (Moving): {os.path.basename(self.img1_path)}")
        print(f"Img2 (Canvas): {os.path.basename(self.img2_path)}")
        
        # Read standard numpy arrays
        self.img1 = cv.imread(self.img1_path)
        self.img2 = cv.imread(self.img2_path)
        
        if self.img1 is None or self.img2 is None:
            print("Error loading images for this pair.")
            return

        # Visually undistort the images for the user if camera params exist
        if os.path.exists('camera_params.json'):
            try:
                with open('camera_params.json', 'r') as f:
                    params = json.load(f)
                K = np.array(params['camera_matrix'])
                dist = np.array(params['dist_coeffs'])
                
                # Get optimal new camera matrix so we don't crop out valid pixels
                h_img, w_img = self.img1.shape[:2]
                new_cameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w_img, h_img), 1, (w_img, h_img))
                
                print("Applying visual lens undistortion to UI images...")
                self.img1 = cv.undistort(self.img1, K, dist, None, new_cameramtx)
                self.img2 = cv.undistort(self.img2, K, dist, None, new_cameramtx)
            except Exception as e:
                print(f"Warning: Failed to apply visual visual undistortion: {e}")
            
        # Create low-res proxies (50% scale) for fast interactive rendering
        self.proxy_scale = 0.5
        self.img1_proxy = cv.resize(self.img1, (0, 0), fx=self.proxy_scale, fy=self.proxy_scale, interpolation=cv.INTER_LINEAR)
        self.img2_proxy = cv.resize(self.img2, (0, 0), fx=self.proxy_scale, fy=self.proxy_scale, interpolation=cv.INTER_LINEAR)
            
        # Upload to GPU memory immediately for rendering operations
        self.u_img1 = cv.UMat(self.img1)
        self.u_img2 = cv.UMat(self.img2)
        self.u_img1_proxy = cv.UMat(self.img1_proxy)
        self.u_img2_proxy = cv.UMat(self.img2_proxy)
            
        self.h1, self.w1 = self.img1.shape[:2]
        self.h2, self.w2 = self.img2.shape[:2]
        
        # Pre-allocate white masks as UMats ONCE per pair (Fixes the 60FPS memory leak)
        self.u_white1 = cv.UMat(np.ones((self.h1, self.w1), dtype=np.uint8)*255)
        self.u_white2 = cv.UMat(np.ones((self.h2, self.w2), dtype=np.uint8)*255)
        
        h1_p, w1_p = self.img1_proxy.shape[:2]
        h2_p, w2_p = self.img2_proxy.shape[:2]
        self.u_white1_proxy = cv.UMat(np.ones((h1_p, w1_p), dtype=np.uint8)*255)
        self.u_white2_proxy = cv.UMat(np.ones((h2_p, w2_p), dtype=np.uint8)*255)
        
        # Original corners of img1 [top-left, bottom-left, bottom-right, top-right]
        self.src_pts = np.float32([[0, 0], [0, self.h1 - 1], [self.w1 - 1, self.h1 - 1], [self.w1 - 1, 0]])
        
        if self.current_idx in self.saved_matrices:
            print("Restoring previously adjusted homography from memory...")
            self.M = self.saved_matrices[self.current_idx].copy()
            self.is_valid_auto_match = self.homography_status.get(self.current_idx, False)
            self.dst_pts = cv.perspectiveTransform(self.src_pts.reshape(-1, 1, 2), self.M).reshape(4, 2)
        else:
            print("Computing initial homography using alignment_utils.py...")
            self.M, self.is_valid_auto_match = self.get_initial_homography()
            self.dst_pts = cv.perspectiveTransform(self.src_pts.reshape(-1, 1, 2), self.M).reshape(4, 2)
            self.saved_matrices[self.current_idx] = self.M.copy()
            self.homography_status[self.current_idx] = self.is_valid_auto_match
            
        # Reset View to fit img2 nicely inside canvas
        self.zoom = min(self.canvas_w / self.w2, self.canvas_h / self.h2) * 0.8
        self.offset_x = (self.canvas_w - self.w2 * self.zoom) / 2
        self.offset_y = (self.canvas_h - self.h2 * self.zoom) / 2
        self.render()
        
    def get_initial_homography(self):
        img1_gray = cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(self.img2, cv.COLOR_BGR2GRAY)
        
        kp1, des1, kp2, des2, good_matches = au.detect_and_match(
            img1_gray, img2_gray, method='SIFT_BF', lowe_ratio=0.7
        )
        
        M, mask = au.compute_homography(kp1, kp2, good_matches, min_match_count=10, img_shape=img1_gray.shape)
        if M is not None:
            print("Initial homography computed successfully.")
            return M, True
        else:
            print("Not enough good matches or homography failed constraints. Using pure translation as fallback.")
            dy = int(self.h2 * 0.85) # Place top of img1 over bottom 15% of img2
            M = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            return M, False
            
    def update_M(self):
        self.M = cv.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.saved_matrices[self.current_idx] = self.M.copy()
        
    def screen_to_world(self, x, y):
        wx = (x - self.offset_x) / self.zoom
        wy = (y - self.offset_y) / self.zoom
        return wx, wy
        
    def mouse_callback(self, event, x, y, flags, param):
        wx, wy = self.screen_to_world(x, y)
        
        if event == cv.EVENT_LBUTTONDOWN:
            distances = np.linalg.norm(self.dst_pts - np.array([wx, wy]), axis=1)
            closest_idx = np.argmin(distances)
            
            # Hitbox in world space scales with zoom
            if distances[closest_idx] < 20 / self.zoom:
                self.dragging_idx = closest_idx
            else:
                self.panning = True
                self.pan_start_x = x
                self.pan_start_y = y
                self.pan_start_offset_x = self.offset_x
                self.pan_start_offset_y = self.offset_y
                
        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging_idx != -1:
                self.dst_pts[self.dragging_idx] = [wx, wy]
                self.update_M()
                self.needs_render = True
            elif self.panning:
                dx = x - self.pan_start_x
                dy = y - self.pan_start_y
                self.offset_x = self.pan_start_offset_x + dx
                self.offset_y = self.pan_start_offset_y + dy
                self.needs_render = True
                
        elif event == cv.EVENT_LBUTTONUP:
            self.dragging_idx = -1
            self.panning = False
            self.render()

        elif event == cv.EVENT_MOUSEWHEEL:
            delta = ctypes.c_int32(flags).value
            if delta > 0:
                zoom_factor = 1.1
            else:
                zoom_factor = 1.0 / 1.1
            
            self.zoom *= zoom_factor
            
            # Zoom to cursor
            self.offset_x = x - wx * self.zoom
            self.offset_y = y - wy * self.zoom
            
            self.render()

    def render(self):
        is_interacting = (self.dragging_idx != -1) or self.panning
        
        # View matrix (World Space to Screen Space)
        S = np.array([
            [self.zoom, 0, self.offset_x],
            [0, self.zoom, self.offset_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        S_M = S @ self.M
        
        render_w = self.canvas_w
        render_h = self.canvas_h
        
        if is_interacting:
            # Render on a drastically smaller canvas to save PCIe transfer times over .get()
            render_w = int(self.canvas_w * self.proxy_scale)
            render_h = int(self.canvas_h * self.proxy_scale)
            
            # P matrix maps WorldSpace to ProxySpace
            P = np.array([[self.proxy_scale, 0, 0], [0, self.proxy_scale, 0], [0, 0, 1]], dtype=np.float32)
            P_inv = np.array([[1.0/self.proxy_scale, 0, 0], [0, 1.0/self.proxy_scale, 0], [0, 0, 1]], dtype=np.float32)
            
            # Transform from ProxySpace -> WorldSpace -> ScreenSpace -> ProxyScreenSpace
            M2_render = P @ S @ P_inv
            M1_render = P @ S_M @ P_inv
            
            u_warped2 = cv.warpPerspective(self.u_img2_proxy, M2_render, (render_w, render_h))
            u_warped1 = cv.warpPerspective(self.u_img1_proxy, M1_render, (render_w, render_h))
            
            u_mask2 = cv.warpPerspective(self.u_white2_proxy, M2_render, (render_w, render_h))
            u_mask1 = cv.warpPerspective(self.u_white1_proxy, M1_render, (render_w, render_h))
        else:
            # Full quality render to massive 1280x720 canvas
            u_warped2 = cv.warpPerspective(self.u_img2, S, (render_w, render_h))
            u_warped1 = cv.warpPerspective(self.u_img1, S_M, (render_w, render_h))
            
            u_mask2 = cv.warpPerspective(self.u_white2, S, (render_w, render_h))
            u_mask1 = cv.warpPerspective(self.u_white1, S_M, (render_w, render_h))
        
        # .get() is the main bottleneck. The smaller the canvas, the faster the PCIe bus transfer.
        warped2 = u_warped2.get()
        warped1 = u_warped1.get()
        mask2 = u_mask2.get()
        mask1 = u_mask1.get()
        
        canvas = np.full((render_h, render_w, 3), 30, dtype=np.uint8)
        
        only2_mask = (mask2 > 0) & ~(mask1 > 0)
        canvas[only2_mask] = warped2[only2_mask]
        
        only1_mask = (mask1 > 0) & ~(mask2 > 0)
        canvas[only1_mask] = warped1[only1_mask]
        
        overlap_mask = (mask1 > 0) & (mask2 > 0)
        
        mode_text = ""
        if self.mode == 0:
            # Alpha
            # Doing alpha blend on GPU
            u_blend = cv.addWeighted(u_warped2, 0.5, u_warped1, 0.5, 0)
            blend = u_blend.get()
            canvas[overlap_mask] = blend[overlap_mask]
            mode_text = "MODE: Alpha Blend (50/50)"
        elif self.mode == 1:
            # Difference on GPU
            u_diff = cv.absdiff(u_warped2, u_warped1)
            diff = u_diff.get()
            canvas[overlap_mask] = diff[overlap_mask]
            mode_text = "MODE: Difference (Darker = Better Alignment)"
        elif self.mode == 2:
            # Anaglyph mixing on CPU arrays
            anaglyph = np.zeros_like(warped2)
            anaglyph[:, :, 0] = warped2[:, :, 0] 
            anaglyph[:, :, 1] = warped2[:, :, 1] 
            anaglyph[:, :, 2] = warped1[:, :, 2] 
            canvas[overlap_mask] = anaglyph[overlap_mask]
            mode_text = "MODE: Anaglyph (Red=Img1, Cyan=Img2)"
        elif self.mode == 3:
            # Edges on GPU
            u_gray1 = cv.cvtColor(u_warped1, cv.COLOR_BGR2GRAY)
            u_gray2 = cv.cvtColor(u_warped2, cv.COLOR_BGR2GRAY)
            
            u_edges1 = cv.Canny(u_gray1, 100, 200)
            u_edges2 = cv.Canny(u_gray2, 100, 200)
            
            edges1 = u_edges1.get()
            edges2 = u_edges2.get()
            
            edge_disp = np.zeros_like(warped2)
            edge_disp[edges2 > 0] = [255, 255, 0]
            edge_disp[edges1 > 0] = [0, 0, 255]
            
            canvas[overlap_mask] = edge_disp[overlap_mask]
            mode_text = "MODE: Edges (Red Edge = Img1, Cyan Edge = Img2)"
            
        # If we rendered dynamically to a small canvas during movement, upscale it back to window size
        if is_interacting:
            canvas = cv.resize(canvas, (self.canvas_w, self.canvas_h), interpolation=cv.INTER_NEAREST)
            
        screen_pts = cv.perspectiveTransform(self.dst_pts.reshape(-1, 1, 2), S).reshape(4, 2)
        pts = np.int32(screen_pts)
        cv.polylines(canvas, [pts], True, (0, 255, 255), 2, cv.LINE_AA)
        
        for pt in pts:
            cv.circle(canvas, tuple(pt), 10, (255, 0, 255), -1) 
            cv.circle(canvas, tuple(pt), 2, (0, 0, 0), -1) 
            
        # Draw status circle (Green = Homography Found, Red = Fallback Defaults)
        status_color = (0, 255, 0) if self.is_valid_auto_match else (0, 0, 255)
        status_text = "AUTO MATCH" if self.is_valid_auto_match else "FALLBACK (FAILED)"
            
        if self.help_visible:
            overlay = canvas.copy()
            cv.rectangle(overlay, (10, 10), (950, 210), (0, 0, 0), -1)
            canvas = cv.addWeighted(overlay, 0.6, canvas, 0.4, 0)
            
            pair_status = f"Pair {self.current_idx + 1}/{len(self.pairs)}"
            cv.putText(canvas, f"{pair_status} | {mode_text}", (20, 45), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv.LINE_AA)
            cv.circle(canvas, (25, 75), 8, status_color, -1)
            cv.putText(canvas, f"Detection: {status_text}", (45, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv.LINE_AA)
            cv.putText(canvas, f"Base (Canvas): {os.path.basename(self.img2_path)}", (20, 105), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv.LINE_AA)
            cv.putText(canvas, f"Overlapping :  {os.path.basename(self.img1_path)}", (20, 125), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 255), 1, cv.LINE_AA)
            cv.putText(canvas, "-> LEFT CLICK + DRAG MAGENTA DOTS to adjust perspective.", (20, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(canvas, "-> LEFT CLICK + DRAG KEYBOARD to Pan | SCROLL to Zoom.", (20, 170), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(canvas, "-> KEYS: [m] mode | [q]/[e] prev/next | [s] save | [h] help | [esc] exit.", (20, 195), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv.LINE_AA)
        else:
            cv.putText(canvas, f"Pair {self.current_idx + 1}/{len(self.pairs)} | {mode_text}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            cv.circle(canvas, (25, 65), 8, status_color, -1)
            cv.putText(canvas, status_text, (45, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv.LINE_AA)
        
        # Fading save status indicator at bottom-left
        if self.save_status is not None:
            elapsed = time.time() - self.save_status_time
            if self.save_status == "saving":
                alpha = 1.0
                label = "Saving..."
                color = (80, 200, 255)
            else:  # "saved"
                fade_start = 0.0
                fade_duration = 1.5
                alpha = max(0.0, 1.0 - max(0.0, elapsed - fade_start) / fade_duration)
                label = "Saved"
                color = (80, 255, 120)
                if alpha <= 0.0:
                    self.save_status = None

            if self.save_status is not None:
                text_y = self.canvas_h - 20
                text_x = 20
                (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                bg = canvas.copy()
                cv.rectangle(bg, (text_x - 8, text_y - th - 8), (text_x + tw + 8, text_y + 8), (0, 0, 0), -1)
                canvas = cv.addWeighted(bg, 0.6 * alpha, canvas, 1.0 - 0.6 * alpha, 0)
                text_color = tuple(int(c * alpha) for c in color)
                cv.putText(canvas, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv.LINE_AA)

        cv.imshow(self.window_name, canvas)

    def run(self):
        print("Starting interactive GUI. Click on the image window and press keys.")
        self.needs_render = True
        while True:
            if self.needs_render:
                self.render()
                self.needs_render = False
            elif self.save_status is not None:
                # Keep rendering while save text is fading
                self.render()
                
            key = cv.waitKeyEx(15)
            if key == -1:
                continue
            
            char_key = key & 0xFF
            
            if char_key == 27: # ESC
                print("Exiting tool.")
                break
            elif char_key == ord('m'):
                self.mode = (self.mode + 1) % self.modes_count
                self.render()
            elif char_key == ord('h'):
                self.help_visible = not self.help_visible
                self.render()
            elif char_key == ord('e') or key == 2555904: # 'e' or Right Arrow
                if self.current_idx < len(self.pairs) - 1:
                    self.current_idx += 1
                    self.load_pair()
            elif char_key == ord('q') or key == 2424832: # 'q' or Left Arrow
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.load_pair()
            elif char_key == ord('s'):
                output_file = "homographies.json"
                self.save_status = "saving"
                self.save_status_time = time.time()
                self.render()  # Show "Saving..." immediately
                cv.waitKey(1)  # Flush the frame

                print(f"Saving Homography Matrix (M) to {output_file}:")
                data = {}
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            pass

                pair_key = f"{os.path.basename(self.img1_path)}|{os.path.basename(self.img2_path)}"
                data[pair_key] = {
                    "M": self.M.tolist(),
                    "projected_corners": self.dst_pts.tolist(),
                    "img1_path": self.img1_path,
                    "img2_path": self.img2_path,
                    "auto_match": self.is_valid_auto_match
                }
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=4)

                self.save_status = "saved"
                self.save_status_time = time.time()
                print(f"Saved successfully! Matrix for {pair_key} added to {output_file}.")
                
        cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Alignment UI for overlapping images.")
    parser.add_argument("--dir", default="images/rgb_27_06_2025", help="Directory containing images to pair automatically")
    parser.add_argument("--img1", help="Path to the first image (the one that moves)")
    parser.add_argument("--img2", help="Path to the second image (static background canvas)")
    parser.add_argument("--start-pair", type=int, default=1, help="The pair index to start the GUI at (e.g. 27)")
    args = parser.parse_args()
    
    pairs = []
    if args.img1 and args.img2:
        pairs.append((args.img1, args.img2))
    elif args.dir and os.path.isdir(args.dir):
        print(f"Scanning directory {args.dir} for image pairs...")
        pairs = find_image_pairs(args.dir)
        print(f"Found {len(pairs)} pairs to process.")
    
    if not pairs:
        print("No valid image pairs found. Please check paths or directory format.")
        exit(1)
        
    start_idx = max(0, min(args.start_pair - 1, len(pairs) - 1))
    ui = ManualAlignmentUI(pairs, start_idx=start_idx)
    ui.run()
