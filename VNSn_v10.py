#!/usr/bin/env python3
"""
vns_lightning_headless.py
Headless VNS pipeline adapted for Lightning AI / headless servers.

Features:
- ArcGIS tile download & stitched orthophoto cache
- Load preplanned route (JSON or light GPX)
- Route-driven candidate tile selection (neighborhood expansion)
- ORB descriptor precompute & retrieval for candidate tiles
- LoFTR fine matching on map crop
- Progressive expansion fallback & simple motion prior
- Process a video, output annotated video + trajectory.json + debug_map.png

IMPORTANT: Do NOT embed your ArcGIS API key in code. Set environment variable ARCGIS_API_KEY.
"""

import os
import sys
import math
import io
import json
import time
import argparse
import tempfile
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from kornia.feature import LoFTR

# ----------------------------
# ========== CONFIG ==========
# ----------------------------
# Default map/video/processing settings (tweak as needed)
DEFAULT_LAT = 30.3682111111111
DEFAULT_LON = 76.4451111111111
DEFAULT_ZOOM = 18
DEFAULT_GRID = 5
TILE_SIZE = 256

MATCH_INTERVAL = 6        # run matching every N frames (increase to reduce GPU load)
TOP_K_TILES = 2
ORB_MAX_FEATURES = 600
MIN_LOFTR_MATCHES = 12
LOFTR_MODEL = "outdoor"

REF_SCALE = 0.6           # for debug map image scaling
PADDING_CROP = 96

# ----------------------------
# ===== Helper functions =====
# ----------------------------
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

# ----------------------------
# map download + stitching
# ----------------------------
def download_and_stitch_map(lat, lon, zoom, grid_size, cache_dir):
    """
    Downloads grid_size x grid_size ArcGIS World Imagery tiles centered at (lat,lon) at zoom level.
    Caches stitched file in cache_dir.
    Returns (PIL.Image RGB, tl_xtile, tl_ytile)
    """
    api_key = os.environ.get("ARCGIS_API_KEY")
    if not api_key:
        raise RuntimeError("ARCGIS_API_KEY environment variable not set")
    center_xtile, center_ytile = deg2num(lat, lon, zoom)
    offset = grid_size // 2
    tl_xtile = center_xtile - offset
    tl_ytile = center_ytile - offset

    cache_file = os.path.join(cache_dir, f"arcgis_cache_z{zoom}_g{grid_size}_{center_xtile}_{center_ytile}.png")
    meta_file = cache_file + ".json"
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            if meta.get("center_xtile") == center_xtile and meta.get("center_ytile") == center_ytile:
                img = Image.open(cache_file).convert("RGB")
                print(f"[MAP] loaded cached stitched map: {cache_file}")
                return img, tl_xtile, tl_ytile
        except Exception:
            pass

    stitched = Image.new("RGB", (grid_size * TILE_SIZE, grid_size * TILE_SIZE))
    headers = {'User-Agent': 'VNS-Headless/1.0'}
    for x in range(grid_size):
        for y in range(grid_size):
            xt = tl_xtile + x
            yt = tl_ytile + y
            url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{zoom}/{yt}/{xt}?token={api_key}"
            try:
                resp = requests.get(url, headers=headers, timeout=12)
                if resp.status_code == 200:
                    tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    stitched.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))
                else:
                    print(f"[MAP] warning: tile {xt},{yt} returned {resp.status_code}")
            except Exception as e:
                print(f"[MAP] tile {xt},{yt} error: {e}")
    stitched.save(cache_file)
    with open(meta_file, "w") as f:
        json.dump({"center_xtile": center_xtile, "center_ytile": center_ytile}, f)
    print(f"[MAP] downloaded & saved stitched map to {cache_file}")
    return stitched, tl_xtile, tl_ytile

# ----------------------------
# tile <-> pixel helpers
# ----------------------------
def tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile, tile_size=TILE_SIZE):
    rel_x = (xt - tl_xtile) * tile_size
    rel_y = (yt - tl_ytile) * tile_size
    return int(rel_x), int(rel_y), int(rel_x + tile_size), int(rel_y + tile_size)

def union_bbox_from_tiles(tile_list, tl_xtile, tl_ytile, tile_size=TILE_SIZE, padding=64, map_w=None, map_h=None):
    x0s, y0s, x1s, y1s = [], [], [], []
    for xt, yt in tile_list:
        x0,y0,x1,y1 = tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile, tile_size)
        x0s.append(x0); y0s.append(y0); x1s.append(x1); y1s.append(y1)
    x0, y0, x1, y1 = min(x0s), min(y0s), max(x1s), max(y1s)
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    if map_w is not None:
        x1 = min(map_w, x1 + padding)
    else:
        x1 = x1 + padding
    if map_h is not None:
        y1 = min(map_h, y1 + padding)
    else:
        y1 = y1 + padding
    return int(x0), int(y0), int(x1), int(y1)

# ----------------------------
# ORB descriptors for retrieval
# ----------------------------
def precompute_tile_descriptors_for_set(map_img, tile_set, tl_xtile, tl_ytile, tile_size=TILE_SIZE, max_features=ORB_MAX_FEATURES):
    orb = cv2.ORB_create(max_features)
    descs = {}
    map_w, map_h = map_img.size
    for xt, yt in tile_set:
        x0,y0,x1,y1 = tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile, tile_size)
        # clamp
        x0 = max(0, min(map_w-1, x0)); y0 = max(0, min(map_h-1, y0))
        x1 = max(0, min(map_w, x1)); y1 = max(0, min(map_h, y1))
        tile = map_img.crop((x0,y0,x1,y1)).convert("RGB")
        tile_np = np.array(tile)
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        kps, desc = orb.detectAndCompute(gray, None)
        descs[(xt,yt)] = desc
    return descs

def find_best_tiles(candidate_tile_descs, query_frame, top_k=TOP_K_TILES):
    orb = cv2.ORB_create(ORB_MAX_FEATURES)
    qgray = cv2.cvtColor(query_frame, cv2.COLOR_RGB2GRAY)
    kq, dq = orb.detectAndCompute(qgray, None)
    if dq is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    scores = []
    for tile_key, desc in candidate_tile_descs.items():
        if desc is None:
            scores.append((tile_key, 0))
            continue
        try:
            matches = bf.match(dq, desc)
            scores.append((tile_key, len(matches)))
        except Exception:
            scores.append((tile_key, 0))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]

# ----------------------------
# LoFTR localizer
# ----------------------------
class Localizer:
    def __init__(self, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[LoFTR] initializing on device: {self.device}")
        self.matcher = LoFTR(pretrained=LOFTR_MODEL).eval().to(self.device)

    def match(self, img_ref, img_query, min_matches=MIN_LOFTR_MATCHES):
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
        gray_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)
        t_ref = torch.from_numpy(gray_ref).unsqueeze(0).unsqueeze(0).float().to(self.device)/255.0
        t_query = torch.from_numpy(gray_query).unsqueeze(0).unsqueeze(0).float().to(self.device)/255.0
        with torch.no_grad():
            corresp = self.matcher({"image0": t_query, "image1": t_ref})
        mkpts0 = corresp['keypoints0'].cpu().numpy()
        mkpts1 = corresp['keypoints1'].cpu().numpy()
        if 'matches0' in corresp:
            matches0 = corresp['matches0'].cpu().numpy()
            mask = matches0 >= 0
            mk0 = mkpts0[mask]
            mk1 = mkpts1[mask]
        else:
            mk0 = mkpts0
            mk1 = mkpts1
        if len(mk0) >= min_matches:
            H, inlier_mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 5.0)
            if H is not None:
                h, w = gray_query.shape
                center = np.array([[[w/2, h/2]]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(center, H)
                return float(dst_pt[0][0][0]), float(dst_pt[0][0][1]), {'inliers': (int(inlier_mask.sum()) if inlier_mask is not None else None), 'matches': len(mk0)}
        return None, None, None

# ----------------------------
# simple motion prior (tile coords)
# ----------------------------
class MotionPrior:
    def __init__(self):
        self.xt = None
        self.yt = None
        self.vx = 0.0
        self.vy = 0.0
        self.last_ts = None

    def update_position(self, xt, yt, ts=None):
        now = ts if ts is not None else time.time()
        if self.xt is None:
            self.xt, self.yt, self.last_ts = xt, yt, now
            return
        dt = now - self.last_ts if self.last_ts is not None else 1e-3
        self.vx = (xt - self.xt) / dt if dt > 0 else 0.0
        self.vy = (yt - self.yt) / dt if dt > 0 else 0.0
        self.xt, self.yt, self.last_ts = xt, yt, now

    def predict(self, dt=0.1):
        if self.xt is None:
            return None, None
        return self.xt + self.vx * dt, self.yt + self.vy * dt

# ----------------------------
# route helpers
# ----------------------------
def load_route_file(route_path):
    """
    Accepts either a JSON with {'waypoints': [[lat,lon], ...]} OR a simple GPX with <trkpt lat="" lon=""/>.
    Returns list of (lat, lon).
    """
    if not os.path.exists(route_path):
        raise FileNotFoundError(route_path)
    if route_path.lower().endswith(".json"):
        with open(route_path, "r") as f:
            data = json.load(f)
        w = data.get("waypoints") or data.get("route") or data.get("points")
        if w is None:
            raise ValueError("JSON route must contain 'waypoints' (list of [lat,lon])")
        return [(float(a[0]), float(a[1])) if isinstance(a, (list,tuple)) else (float(a['lat']), float(a['lon'])) for a in w]
    else:
        # light GPX parse
        txt = open(route_path, "r").read()
        import re
        pts = []
        for m in re.finditer(r'<trkpt\\s+lat=\"([0-9\\.\\-]+)\"\\s+lon=\"([0-9\\.\\-]+)\"', txt):
            pts.append((float(m.group(1)), float(m.group(2))))
        if not pts:
            raise ValueError("GPX parse found 0 points")
        return pts

def waypoints_to_tiles(waypoints, zoom):
    tiles = set()
    for lat, lon in waypoints:
        x,y = deg2num(lat, lon, zoom)
        tiles.add((x,y))
    return sorted(list(tiles))

def tile_neighborhood(tile_set, radius=1):
    nbrs = set()
    for xt, yt in tile_set:
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nbrs.add((xt+dx, yt+dy))
    return sorted(list(nbrs))

# ----------------------------
# draw debug map image
# ----------------------------
def render_debug_map(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, last_match_stitched=None, out_path=None):
    img = map_img.copy()
    draw = ImageDraw.Draw(img)
    w,h = img.size
    # draw route tiles centers as green dots
    for xt,yt in route_tiles:
        cx = ((xt - tl_xtile) * TILE_SIZE + TILE_SIZE/2)
        cy = ((yt - tl_ytile) * TILE_SIZE + TILE_SIZE/2)
        r = 6
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(0,255,0))
    # draw route lines
    if len(route_tiles) > 1:
        pts = [((xt - tl_xtile) * TILE_SIZE + TILE_SIZE/2, (yt - tl_ytile) * TILE_SIZE + TILE_SIZE/2) for xt,yt in route_tiles]
        draw.line(pts, fill=(0,255,0), width=3)
    # candidate tile boxes
    for xt,yt in candidate_tiles:
        x0,y0,x1,y1 = tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile)
        draw.rectangle((x0,y0,x1,y1), outline=(255,153,0), width=2)
    # last match
    if last_match_stitched is not None:
        mx,my = last_match_stitched
        r = 10
        draw.ellipse((mx-r,my-r,mx+r,my+r), fill=(255,0,0), outline=(255,255,255), width=2)
        draw.text((mx+12, my-12), "DRONE POS", fill=(255,255,255))
    if out_path is not None:
        img.save(out_path)
    return img

# ----------------------------
# main processing pipeline
# ----------------------------
def run_headless(
    video_path,
    route_path,
    out_dir,
    center_lat=None,
    center_lon=None,
    zoom=DEFAULT_ZOOM,
    grid_size=DEFAULT_GRID,
    search_radius=1,
    match_interval=MATCH_INTERVAL,
    top_k_tiles=TOP_K_TILES
):
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = out_dir

    print("[RUN] loading route...")
    waypoints = load_route_file(route_path)
    print(f"[RUN] loaded {len(waypoints)} waypoints")
    route_tiles = waypoints_to_tiles(waypoints, zoom)
    if not route_tiles:
        raise RuntimeError("route produced no tiles")

    # center map on route mean if no explicit center provided
    if center_lat is None or center_lon is None:
        mean_lat = sum([p[0] for p in waypoints]) / len(waypoints)
        mean_lon = sum([p[1] for p in waypoints]) / len(waypoints)
        center_lat, center_lon = mean_lat, mean_lon

    print("[RUN] downloading stitched map...")
    map_img, tl_xtile, tl_ytile = download_and_stitch_map(center_lat, center_lon, zoom, grid_size, cache_dir)
    map_w, map_h = map_img.size
    print(f"[RUN] map size: {map_w} x {map_h}; top-left tile: {tl_xtile},{tl_ytile}")

    # candidate tiles (neighborhood of route)
    candidate_tiles = tile_neighborhood(route_tiles, radius=search_radius)
    print(f"[RUN] candidate tiles count (radius={search_radius}): {len(candidate_tiles)}")

    # precompute descriptors for candidate set
    print("[RUN] precomputing ORB descriptors for candidate tiles...")
    tile_descs = precompute_tile_descriptors_for_set(map_img, candidate_tiles, tl_xtile, tl_ytile)
    print("[RUN] done precomputing descriptors")

    # initialize localizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    localizer = Localizer(device=device)
    motion = MotionPrior()

    # open video
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(out_dir, "output_annotated.mp4")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    print(f"[RUN] video opened: {video_path} (fps={fps}, size={width}x{height})")
    frame_idx = 0
    last_match_stitched = None
    trajectory = []
    progressive_fail_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # matching step
        if (frame_idx % match_interval == 0):
            # ensure tile_descs cover candidate tiles (in case search radius changed)
            missing = [t for t in candidate_tiles if t not in tile_descs]
            if missing:
                print("[RUN] computing descriptors for missing tiles", len(missing))
                tile_descs.update(precompute_tile_descriptors_for_set(map_img, missing, tl_xtile, tl_ytile))
            # find top K tiles using ORB retrieval
            top_tiles = find_best_tiles({k: tile_descs[k] for k in candidate_tiles}, frame_rgb, top_k=top_k_tiles)
            if len(top_tiles) == 0:
                progressive_fail_count += 1
                trajectory.append({"frame": frame_idx, "matched": False})
                # progressive expansion fallback
                if progressive_fail_count > 8:
                    search_radius = min(5, search_radius + 1)
                    print(f"[RUN] expanding search radius to {search_radius}")
                    candidate_tiles = tile_neighborhood(route_tiles, radius=search_radius)
                    tile_descs = precompute_tile_descriptors_for_set(map_img, candidate_tiles, tl_xtile, tl_ytile)
                    progressive_fail_count = 0
            else:
                # build union bbox around these top tiles and crop
                x0,y0,x1,y1 = union_bbox_from_tiles(top_tiles, tl_xtile, tl_ytile, TILE_SIZE, padding=PADDING_CROP, map_w=map_w, map_h=map_h)
                ref_crop = map_img.crop((x0,y0,x1,y1)).convert("RGB")
                ref_np = np.array(ref_crop)
                # run LoFTR fine match
                mx_rel, my_rel, info = localizer.match(ref_np, frame_rgb)
                if mx_rel is not None:
                    stitched_x = x0 + mx_rel
                    stitched_y = y0 + my_rel
                    last_match_stitched = (stitched_x, stitched_y)
                    # compute lat/lon using fractional tile indices
                    real_x = stitched_x / TILE_SIZE
                    real_y = stitched_y / TILE_SIZE
                    lat, lon = num2deg(tl_xtile + real_x, tl_ytile + real_y, zoom)
                    confidence = info.get("inliers") if info else None
                    trajectory.append({"frame": frame_idx, "matched": True, "lat": lat, "lon": lon, "inliers": confidence, "top_tiles": top_tiles})
                    # update motion prior
                    tx = tl_xtile + int(stitched_x // TILE_SIZE)
                    ty = tl_ytile + int(stitched_y // TILE_SIZE)
                    motion.update_position(tx, ty)
                    progressive_fail_count = 0
                else:
                    progressive_fail_count += 1
                    trajectory.append({"frame": frame_idx, "matched": False, "reason": "loftr_fail"})
        else:
            # no matching this frame; optionally predict with motion prior
            predicted = motion.predict(dt=0.1)
            if predicted[0] is not None:
                # no lat/lon computed; still push predicted tile coords
                trajectory.append({"frame": frame_idx, "matched": False, "predicted_tile": predicted})
            else:
                trajectory.append({"frame": frame_idx, "matched": False})

        # annotate frame for output video
        annotated = frame.copy()
        # write last known location text
        if last_match_stitched is not None and len(trajectory) > 0 and trajectory[-1].get("matched"):
            lat = trajectory[-1]["lat"]; lon = trajectory[-1]["lon"]
            txt = f"LAT: {lat:.6f} LON: {lon:.6f}"
        else:
            txt = "GPS: STANDBY"
        # overlay a black rectangle for readability
        cv2.rectangle(annotated, (8,8), (320,72), (0,0,0), thickness=-1)
        cv2.putText(annotated, txt, (16,44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        writer.write(annotated)

    # finish
    cap.release()
    writer.release()

    # render debug map (large image)
    debug_map_path = os.path.join(out_dir, "debug_map.png")
    render_debug_map(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, last_match_stitched, out_path=debug_map_path)

    traj_path = os.path.join(out_dir, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"[DONE] annotated video: {out_video_path}")
    print(f"[DONE] trajectory JSON: {traj_path}")
    print(f"[DONE] debug map: {debug_map_path}")

# ----------------------------
# CLI entry
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Headless VNS for Lightning AI")
    p.add_argument("--video", required=True, help="Path to input video (mp4)")
    p.add_argument("--route", required=True, help="Path to route JSON or GPX")
    p.add_argument("--outdir", default="/content/data/vns_out", help="Output directory (persistent)")
    p.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    p.add_argument("--grid", type=int, default=DEFAULT_GRID)
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--match_interval", type=int, default=MATCH_INTERVAL)
    p.add_argument("--topk", type=int, default=TOP_K_TILES)
    p.add_argument("--center_lat", type=float, default=None)
    p.add_argument("--center_lon", type=float, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_headless(
        video_path=args.video,
        route_path=args.route,
        out_dir=args.outdir,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        zoom=args.zoom,
        grid_size=args.grid,
        search_radius=args.radius,
        match_interval=args.match_interval,
        top_k_tiles=args.topk
    )
