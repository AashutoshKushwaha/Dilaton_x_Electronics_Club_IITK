import os
import cv2
import yaml
import numpy as np

# -------------------------
# Load config
# -------------------------
CONFIG_PATH = "configs/config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

tile_size = cfg["map_tile_size"]
alt_min = cfg["drone_altitude_min"]
alt_max = cfg["drone_altitude_max"]

tiles_dir = "data/satellite/raw_tiles/tiles"
output_dir = "data/satellite/drone_like_views"

os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Parameters (simple & safe)
# -------------------------
altitude_steps = [alt_min, (alt_min + alt_max) // 2, alt_max]
rotation_angles = [0, 90, 180, 270]

print("[INFO] Generating drone-like views")

# -------------------------
# Helper: scale image
# -------------------------
def scale_image(img, scale):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized

# -------------------------
# Process each tile
# -------------------------
tile_files = sorted(os.listdir(tiles_dir))

for tile_name in tile_files:
    tile_path = os.path.join(tiles_dir, tile_name)
    tile_img = cv2.imread(tile_path)

    if tile_img is None:
        continue

    tile_id = os.path.splitext(tile_name)[0]
    tile_out_dir = os.path.join(output_dir, tile_id)
    os.makedirs(tile_out_dir, exist_ok=True)

    for alt in altitude_steps:
        # lower altitude → bigger scale
        scale = alt_min / alt

        scaled = scale_image(tile_img, scale)

        # center crop back to tile size
        h, w = scaled.shape[:2]
        if h < tile_size or w < tile_size:
            continue

        y0 = (h - tile_size) // 2
        x0 = (w - tile_size) // 2
        crop = scaled[y0:y0+tile_size, x0:x0+tile_size]

        for angle in rotation_angles:
            M = cv2.getRotationMatrix2D(
                (tile_size // 2, tile_size // 2),
                angle,
                1.0
            )
            rotated = cv2.warpAffine(
                crop, M, (tile_size, tile_size)
            )

            out_name = f"view_s{alt}_r{angle}.png"
            out_path = os.path.join(tile_out_dir, out_name)
            cv2.imwrite(out_path, rotated)

print("✅ STEP 3 COMPLETE: Drone-like views generated")
print("Saved in:", output_dir)
