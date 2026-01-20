import cv2
import numpy as np
import torch
import os
from kornia.feature import LoFTR

# =================================================
# PERFORMANCE / ACCURACY CONFIG  (EDIT ONLY THIS)
# =================================================

MATCH_EVERY_N_FRAMES = 30     # ↑ smoother, ↓ more accurate
TOP_K = 2                    # number of LoFTR candidates
MATCH_FRAME_WIDTH = 640      # ↓ faster, ↑ more accurate
MATCH_FRAME_HEIGHT = 360
FORCE_CPU = True             # True = more stable on laptops

# =================================================

VIDEO_PATH = "data/drone_video/flight.mp4"
MAP_PATH = "data/satellite/raw_tiles/map.png"
DESCRIPTORS_PATH = "outputs/global_descriptors/descriptors.npy"
IMAGE_PATHS_FILE = "outputs/global_descriptors/image_paths.txt"

TILE_SIZE = 256
DEBUG_DIR = "outputs/debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# Device selection
# -------------------------
if FORCE_CPU:
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load map descriptors
# -------------------------
map_descs = np.load(DESCRIPTORS_PATH)
with open(IMAGE_PATHS_FILE, "r") as f:
    map_img_paths = [l.strip() for l in f.readlines()]

# -------------------------
# Descriptor function
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
# Load LoFTR
# -------------------------
matcher = LoFTR(pretrained="outdoor").to(device).eval()

# -------------------------
# Video loop
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0
current_location = None
full_map = cv2.imread(MAP_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    display_frame = frame.copy()

    # 🔽 Resize frame ONLY for matching (big speed gain)
    frame_small = cv2.resize(
        frame,
        (MATCH_FRAME_WIDTH, MATCH_FRAME_HEIGHT)
    )

    # ------------------------------------------------
    # HEAVY MATCHING (RUN RARELY)
    # ------------------------------------------------
    if frame_id % MATCH_EVERY_N_FRAMES == 0:
        drone_desc = compute_global_descriptor(frame_small)
        scores = map_descs @ drone_desc
        top_ids = np.argsort(scores)[-TOP_K:][::-1]

        best_score = -1
        best_H = None
        best_tile = None

        for idx in top_ids:
            map_img = cv2.imread(map_img_paths[idx])
            if map_img is None:
                continue

            gray1 = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

            t1 = torch.from_numpy(gray1)[None, None].float().to(device) / 255.0
            t2 = torch.from_numpy(gray2)[None, None].float().to(device) / 255.0

            with torch.no_grad():
                out = matcher({"image0": t1, "image1": t2})

            if len(out["keypoints0"]) < 10:
                continue

            pts1 = out["keypoints0"].cpu().numpy()
            pts2 = out["keypoints1"].cpu().numpy()

            H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
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

            current_location = (
                int(c * TILE_SIZE + px),
                int(r * TILE_SIZE + py)
            )

    # ------------------------------------------------
    # VISUALIZATION (LIGHTWEIGHT)
    # ------------------------------------------------
    if frame_id % MATCH_EVERY_N_FRAMES == 0 and current_location is not None:
        map_vis = full_map.copy()
        cv2.circle(map_vis, current_location, 8, (0, 0, 255), -1)
        cv2.imshow("Map", map_vis)

    cv2.imshow("Drone Video", display_frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

