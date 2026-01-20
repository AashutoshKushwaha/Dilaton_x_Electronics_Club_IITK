import cv2
import numpy as np
import torch
from kornia.feature import LoFTR
import os

# -------------------------
# INPUTS (from STEP 6 result)
# -------------------------
BEST_VIEW_PATH = "data/satellite/drone_like_views/tile_002_002/view_s40_r270.png"
TILE_ROW = 2   # from tile_002_002
TILE_COL = 2

VIDEO_PATH = "data/drone_video/flight.mp4"
MAP_PATH = "data/satellite/raw_tiles/map.png"

TILE_SIZE = 256
DEBUG_DIR = "outputs/debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# Load LoFTR
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
matcher = LoFTR(pretrained="outdoor").to(device).eval()

# -------------------------
# Load drone frame
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, drone_frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read drone frame")

drone_gray = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2GRAY)
drone_tensor = torch.from_numpy(drone_gray)[None, None].float().to(device) / 255.0

# -------------------------
# Load best map view
# -------------------------
map_view = cv2.imread(BEST_VIEW_PATH)
map_gray = cv2.cvtColor(map_view, cv2.COLOR_BGR2GRAY)
map_tensor = torch.from_numpy(map_gray)[None, None].float().to(device) / 255.0

# -------------------------
# Run LoFTR again (for homography)
# -------------------------
with torch.no_grad():
    out = matcher({
        "image0": drone_tensor,
        "image1": map_tensor
    })

pts_drone = out["keypoints0"].cpu().numpy()
pts_map = out["keypoints1"].cpu().numpy()

if len(pts_drone) < 10:
    raise RuntimeError("Not enough matches for homography")

# -------------------------
# Compute homography
# -------------------------
H, mask = cv2.findHomography(pts_drone, pts_map, cv2.RANSAC, 5.0)

if H is None:
    raise RuntimeError("Homography failed")

# -------------------------
# Project drone center
# -------------------------
h, w = drone_frame.shape[:2]
drone_center = np.array([[w / 2, h / 2]], dtype=np.float32).reshape(-1, 1, 2)

projected = cv2.perspectiveTransform(drone_center, H)
px, py = projected[0][0]

# -------------------------
# Convert to global map coordinates
# -------------------------
global_x = TILE_COL * TILE_SIZE + px
global_y = TILE_ROW * TILE_SIZE + py

print("\n✅ DRONE LOCATION ESTIMATED")
print(f"Map pixel X: {int(global_x)}")
print(f"Map pixel Y: {int(global_y)}")

# -------------------------
# Visualize on full map
# -------------------------
full_map = cv2.imread(MAP_PATH)
cv2.circle(full_map, (int(global_x), int(global_y)), 10, (0, 0, 255), -1)

cv2.imwrite(os.path.join(DEBUG_DIR, "drone_on_map.png"), full_map)

print("Saved visualization to outputs/debug_images/drone_on_map.png")
