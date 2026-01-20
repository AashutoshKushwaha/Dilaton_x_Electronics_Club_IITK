import os
import cv2
import numpy as np

# -------------------------
# Paths
# -------------------------
INPUT_DIR = "data/satellite/drone_like_views"
OUTPUT_DIR = "outputs/global_descriptors"

os.makedirs(OUTPUT_DIR, exist_ok=True)

descriptor_list = []
image_paths = []

print("[INFO] Building global descriptors")

# -------------------------
# Helper: compute descriptor
# -------------------------
def compute_global_descriptor(img):
    """
    Very lightweight descriptor:
    - color histogram (HSV)
    - edge histogram
    """
    # Resize for consistency
    img = cv2.resize(img, (128, 128))

    # ----- Color histogram -----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])

    # ----- Edge histogram -----
    edges = cv2.Canny(img, 100, 200)
    edge_hist = np.histogram(edges, bins=16, range=(0, 256))[0]

    # Combine & normalize
    descriptor = np.concatenate([
        hist_h.flatten(),
        hist_s.flatten(),
        edge_hist.flatten()
    ]).astype(np.float32)

    descriptor /= (np.linalg.norm(descriptor) + 1e-6)
    return descriptor

# -------------------------
# Process all drone-like views
# -------------------------
for tile_id in os.listdir(INPUT_DIR):
    tile_folder = os.path.join(INPUT_DIR, tile_id)

    if not os.path.isdir(tile_folder):
        continue

    for img_name in os.listdir(tile_folder):
        img_path = os.path.join(tile_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        desc = compute_global_descriptor(img)

        descriptor_list.append(desc)
        image_paths.append(img_path)

# -------------------------
# Save descriptors
# -------------------------
descriptor_array = np.vstack(descriptor_list)

np.save(os.path.join(OUTPUT_DIR, "descriptors.npy"), descriptor_array)

with open(os.path.join(OUTPUT_DIR, "image_paths.txt"), "w") as f:
    for p in image_paths:
        f.write(p + "\n")

print(f"[DONE] Built descriptors for {len(image_paths)} images")
print("Saved to:", OUTPUT_DIR)
