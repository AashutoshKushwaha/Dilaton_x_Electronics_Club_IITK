import cv2
import torch
import numpy as np
import os
from kornia.feature import LoFTR

# -------------------------
# Paths
# -------------------------
VIDEO_PATH = "data/drone_video/flight.mp4"
CANDIDATE_LIST = "outputs/global_descriptors/image_paths.txt"
DEBUG_DIR = "outputs/debug_images"

os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# Load top-5 candidates from STEP 5
# -------------------------
# (reuse the printed candidates manually for now)
TOP_CANDIDATES = [
    "data/satellite/drone_like_views/tile_002_002/view_s40_r180.png",
    "data/satellite/drone_like_views/tile_002_002/view_s40_r90.png",
    "data/satellite/drone_like_views/tile_002_002/view_s40_r270.png",
    "data/satellite/drone_like_views/tile_002_002/view_s40_r0.png",
    "data/satellite/drone_like_views/tile_003_002/view_s40_r90.png",
]

# -------------------------
# Load LoFTR
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
matcher = LoFTR(pretrained="outdoor").to(device).eval()

# -------------------------
# Read drone frame
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read drone frame")

drone_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Convert to tensor
drone_tensor = torch.from_numpy(drone_gray)[None, None].float().to(device) / 255.0

# -------------------------
# Match against candidates
# -------------------------
best_score = -1
best_candidate = None

print("\n[INFO] Running LoFTR on candidates...\n")

for idx, cand_path in enumerate(TOP_CANDIDATES):
    cand_img = cv2.imread(cand_path)
    if cand_img is None:
        continue

    cand_gray = cv2.cvtColor(cand_img, cv2.COLOR_BGR2GRAY)
    cand_tensor = torch.from_numpy(cand_gray)[None, None].float().to(device) / 255.0

    with torch.no_grad():
        out = matcher({
            "image0": drone_tensor,
            "image1": cand_tensor
        })

    mkpts0 = out["keypoints0"].cpu().numpy()
    mkpts1 = out["keypoints1"].cpu().numpy()

    score = len(mkpts0)
    print(f"Candidate {idx+1}: {cand_path}")
    print(f"  Matches found: {score}")

    # Draw matches for debugging
    debug_img = cv2.drawMatches(
        frame, [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts0],
        cand_img, [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts1],
        matches1to2=[cv2.DMatch(i, i, 0) for i in range(len(mkpts0))],
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        os.path.join(DEBUG_DIR, f"loftr_match_{idx+1}.png"),
        debug_img
    )

    if score > best_score:
        best_score = score
        best_candidate = cand_path

print("\n✅ BEST MATCH FOUND:")
print(best_candidate)
print("Total matches:", best_score)
