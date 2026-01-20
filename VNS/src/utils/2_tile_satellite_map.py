import os
import cv2
import yaml

# -------------------------
# Load config
# -------------------------
CONFIG_PATH = "configs/config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

satellite_dir = cfg["satellite_map_dir"]
tile_size = cfg["map_tile_size"]

# Output directory for tiles
tiles_dir = os.path.join(satellite_dir, "tiles")
os.makedirs(tiles_dir, exist_ok=True)

print("[INFO] Satellite tiling started")

# -------------------------
# Load satellite image
# -------------------------
sat_files = os.listdir(satellite_dir)
sat_files = [f for f in sat_files if f.endswith((".png", ".jpg", ".jpeg"))]

if len(sat_files) == 0:
    raise RuntimeError("No satellite image found")

sat_path = os.path.join(satellite_dir, sat_files[0])
sat_img = cv2.imread(sat_path)

if sat_img is None:
    raise RuntimeError("Failed to load satellite image")

H, W, _ = sat_img.shape
print(f"[OK] Satellite image loaded: {W}x{H}")

# -------------------------
# Tile the image
# -------------------------
tile_count = 0

for y in range(0, H - tile_size + 1, tile_size):
    for x in range(0, W - tile_size + 1, tile_size):

        tile = sat_img[y:y+tile_size, x:x+tile_size]

        tile_name = f"tile_{y//tile_size:03d}_{x//tile_size:03d}.png"
        tile_path = os.path.join(tiles_dir, tile_name)

        cv2.imwrite(tile_path, tile)
        tile_count += 1

print(f"[DONE] Created {tile_count} satellite tiles")
print("Tiles saved in:", tiles_dir)
