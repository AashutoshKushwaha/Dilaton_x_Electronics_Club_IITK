import cv2
import numpy as np
import torch
import os
import tkinter as tk
from PIL import Image, ImageTk
from kornia.feature import LoFTR

# =================================================
# PERFORMANCE CONFIG
# =================================================
MATCH_EVERY_N_FRAMES = 30
TOP_K = 2
MATCH_W, MATCH_H = 640, 360
FORCE_CPU = True
# =================================================

# =================================================
# UI CONFIG (CHANGE ONLY THIS FOR SIZE)
# =================================================
WINDOW_W, WINDOW_H = 900, 480
PANEL_SIZE = 400   # map & video size (square)
# =================================================

VIDEO_PATH = "data/drone_video/flight.mp4"
MAP_PATH = "data/satellite/raw_tiles/map.png"
DESCRIPTORS_PATH = "outputs/global_descriptors/descriptors.npy"
IMAGE_PATHS_FILE = "outputs/global_descriptors/image_paths.txt"
TILE_SIZE = 256

# -------------------------
# Device
# -------------------------
device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load descriptors
# -------------------------
map_descs = np.load(DESCRIPTORS_PATH)
with open(IMAGE_PATHS_FILE, "r") as f:
    map_img_paths = [l.strip() for l in f.readlines()]

# -------------------------
# Descriptor
# -------------------------
def compute_global_descriptor(img):
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    h2 = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    edges = cv2.Canny(img, 100, 200)
    h3 = np.histogram(edges, bins=16, range=(0, 256))[0]
    desc = np.concatenate([h1.flatten(), h2.flatten(), h3.flatten()]).astype(np.float32)
    return desc / (np.linalg.norm(desc) + 1e-6)

# -------------------------
# LoFTR
# -------------------------
matcher = LoFTR(pretrained="outdoor").to(device).eval()

# -------------------------
# UI CLASS
# -------------------------
class VNSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VNS – Live Visual Navigation")

        # ---- Center window ----
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - WINDOW_W) // 2
        y = (screen_h - WINDOW_H) // 2
        root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")
        root.resizable(False, False)

        self.tracking = False
        self.frame_id = 0
        self.current_location = None

        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.full_map = cv2.imread(MAP_PATH)

        # ---- Layout ----
        main = tk.Frame(root)
        main.pack(pady=5)

        self.map_label = tk.Label(main)
        self.map_label.grid(row=0, column=0, padx=5)

        self.video_label = tk.Label(main)
        self.video_label.grid(row=0, column=1, padx=5)

        self.btn = tk.Button(
            root, text="START TRACKING",
            font=("Arial", 12, "bold"),
            bg="green", fg="white",
            command=self.toggle_tracking,
            width=20
        )
        self.btn.pack(pady=5)

        self.update_loop()

    def toggle_tracking(self):
        self.tracking = not self.tracking
        if self.tracking:
            self.btn.config(text="STOP TRACKING", bg="red")
        else:
            self.btn.config(text="START TRACKING", bg="green")

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        self.frame_id += 1

        # ---------- Matching ----------
        if self.tracking and self.frame_id % MATCH_EVERY_N_FRAMES == 0:
            frame_small = cv2.resize(frame, (MATCH_W, MATCH_H))
            drone_desc = compute_global_descriptor(frame_small)
            scores = map_descs @ drone_desc
            top_ids = np.argsort(scores)[-TOP_K:][::-1]

            best_score, best_H, best_tile = -1, None, None

            for idx in top_ids:
                map_img = cv2.imread(map_img_paths[idx])
                if map_img is None:
                    continue

                g1 = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

                t1 = torch.from_numpy(g1)[None, None].float().to(device) / 255
                t2 = torch.from_numpy(g2)[None, None].float().to(device) / 255

                with torch.no_grad():
                    out = matcher({"image0": t1, "image1": t2})

                if len(out["keypoints0"]) < 10:
                    continue

                pts1 = out["keypoints0"].cpu().numpy()
                pts2 = out["keypoints1"].cpu().numpy()

                H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5)
                if H is None:
                    continue

                if len(pts1) > best_score:
                    best_score = len(pts1)
                    best_H = H
                    best_tile = map_img_paths[idx]

            if best_H is not None:
                h, w = frame_small.shape[:2]
                center = np.array([[w / 2, h / 2]], dtype=np.float32).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(center, best_H)
                px, py = proj[0][0]

                tile_name = os.path.basename(os.path.dirname(best_tile))
                r, c = map(int, tile_name.split("_")[1:3])
                self.current_location = (int(c * TILE_SIZE + px), int(r * TILE_SIZE + py))

        # ---------- Map display ----------
        map_vis = self.full_map.copy()
        if self.current_location:
            cv2.circle(map_vis, self.current_location, 8, (0, 0, 255), -1)

        map_vis = cv2.resize(map_vis, (PANEL_SIZE, PANEL_SIZE))
        map_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(map_vis, cv2.COLOR_BGR2RGB)))
        self.map_label.configure(image=map_img)
        self.map_label.image = map_img

        # ---------- Video display ----------
        video_vis = cv2.resize(frame, (PANEL_SIZE, PANEL_SIZE))
        vid_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(video_vis, cv2.COLOR_BGR2RGB)))
        self.video_label.configure(image=vid_img)
        self.video_label.image = vid_img

        self.root.after(30, self.update_loop)

# -------------------------
# RUN
# -------------------------
root = tk.Tk()
app = VNSApp(root)
root.mainloop()
