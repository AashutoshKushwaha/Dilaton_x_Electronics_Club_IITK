import os
import math
from PIL import Image
from main import deg2num, download_and_stitch_map

# CONFIG (match your run)
LAT = None      # auto from route
LON = None
ZOOM = 18
GRID = 5
TILE_SIZE = 256

OUT_DIR = "tiles"
os.makedirs(OUT_DIR, exist_ok=True)

# Load stitched map
map_img, tl_xtile, tl_ytile = download_and_stitch_map(
    lat=30.3682111111111,
    lon=76.4451111111111,
    zoom=ZOOM,
    grid_size=GRID,
    cache_dir="."
)

w, h = map_img.size
nx = w // TILE_SIZE
ny = h // TILE_SIZE

for ix in range(nx):
    for iy in range(ny):
        x0 = ix * TILE_SIZE
        y0 = iy * TILE_SIZE
        tile = map_img.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE))
        xt = tl_xtile + ix
        yt = tl_ytile + iy
        tile.save(f"{OUT_DIR}/tile_{xt}_{yt}.png")

print(f"[OK] Saved {nx*ny} tiles in {OUT_DIR}/")
