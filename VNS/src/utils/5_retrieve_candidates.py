import cv2
import numpy as np
import os

# -------------------------
# Paths
# -------------------------
VIDEO_PATH = "data/drone_video/flight.mp4"
DESCRIPTOR_PATH = "outputs/global_descriptors/descriptors.npy"
IMAGE_LIST_PATH = "outputs/global_descriptors/image_paths.txt"
DEBUG_DIR = "outputs/debug_images"

os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# Load map descriptors
# -------------------------
map_descriptors = np.load(DESCRIPTOR_PATH)

with open(IMAGE_LIST_PATH, "r") as f:
    map_image_paths = [line.strip() for line in f.readlines()]

print("[INFO] Loaded map descriptors")

# -------------------------
# Descriptor function (same as STEP 4)
# -------------------------
def compute_global_descriptor(img):
    img = cv2.resize(img, (128, 128))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])

    edges = cv2.Canny(img, 100, 200)
    edge_hist = np.histogram(edges, bins=16, range=(0, 256))[0]

    desc = np.concatenate([
        hist_h.flatten(),
        hist_s.flatten(),
        edge_hist.flatten()
    ]).astype(np.float32)

    desc /= (np.linalg.norm(desc) + 1e-6)
    return desc

# -------------------------
# Read one frame from video
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame from video")

cv2.imwrite(os.path.join(DEBUG_DIR, "drone_frame.png"), frame)

print("[OK] Drone frame extracted")

# -------------------------
# Compute descriptor
# -------------------------
drone_desc = compute_global_descriptor(frame)

# -------------------------
# Compute similarity (cosine)
# -------------------------
scores = map_descriptors @ drone_desc
top_k = 5
top_indices = np.argsort(scores)[-top_k:][::-1]

print("\nTop candidate map views:")
for rank, idx in enumerate(top_indices):
    print(f"{rank+1}. {map_image_paths[idx]}")

    # Save visual comparison
    img = cv2.imread(map_image_paths[idx])
    out_path = os.path.join(DEBUG_DIR, f"candidate_{rank+1}.png")
    cv2.imwrite(out_path, img)

print("\n✅ STEP 5 COMPLETE: Candidates retrieved")
