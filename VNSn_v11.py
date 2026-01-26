#!/usr/bin/env python3
"""
vns_lightning_composite.py

Headless VNS pipeline that:
- downloads ArcGIS tiles (cached),
- loads a preplanned route (route.json or GPX),
- uses route neighborhood + waypoint-weighted ORB retrieval to pick candidate tiles,
- runs LoFTR for fine matching,
- builds an output video with LEFT = camera feed, RIGHT = virtual map with trail of predicted positions,
- outputs trajectory.json and debug_map.png.

IMPORTANT:
- Set ARCGIS_API_KEY as environment variable (do not hardcode).
- Install dependencies (see instructions below).
"""
import os
import sys
import math
import io
import json
import time
import argparse
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
from kornia.feature import LoFTR

# ---------------- Config (tweak as needed) ----------------
DEFAULT_ZOOM = 18
DEFAULT_GRID = 5
TILE_SIZE = 256

MATCH_INTERVAL = 6
TOP_K_TILES = 2
ORB_MAX_FEATURES = 600
MIN_LOFTR_MATCHES = 12
LOFTR_MODEL = "outdoor"

PADDING_CROP = 96
REF_MAP_DISPLAY_HEIGHT = 480   # map display height in pixels (will match video height)

# Trail settings
TRAIL_MAX_POINTS = 200

# ---------------- Helper functions ----------------
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

# ---------------- Map download & cache ----------------
def download_and_stitch_map(lat, lon, zoom, grid_size, cache_dir):
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
                print("[MAP] loaded cached stitched map")
                return img, tl_xtile, tl_ytile
        except Exception:
            pass
    stitched = Image.new("RGB", (grid_size * TILE_SIZE, grid_size * TILE_SIZE))
    headers = {'User-Agent': 'VNS-Composite/1.0'}
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
                    print(f"[MAP] tile {xt},{yt} status {resp.status_code}")
            except Exception as e:
                print(f"[MAP] tile error {xt},{yt} : {e}")
    stitched.save(cache_file)
    with open(meta_file, "w") as f:
        json.dump({"center_xtile": center_xtile, "center_ytile": center_ytile}, f)
    print("[MAP] downloaded & cached stitched map")
    return stitched, tl_xtile, tl_ytile

# ---------------- ORB retrieval (with weighting) ----------------
def precompute_tile_descriptors_for_set(map_img, tile_set, tl_xtile, tl_ytile, tile_size=TILE_SIZE, max_features=ORB_MAX_FEATURES):
    orb = cv2.ORB_create(max_features)
    descs = {}
    map_w, map_h = map_img.size
    for xt, yt in tile_set:
        x0,y0,x1,y1 = tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile, tile_size)
        x0 = max(0, min(map_w-1, x0)); y0 = max(0, min(map_h-1, y0))
        x1 = max(0, min(map_w, x1)); y1 = max(0, min(map_h, y1))
        tile = map_img.crop((x0,y0,x1,y1)).convert("RGB")
        tile_np = np.array(tile)
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        kps, desc = orb.detectAndCompute(gray, None)
        descs[(xt,yt)] = desc
    return descs

def find_best_tiles(candidate_tile_descs, query_frame, top_k=TOP_K_TILES, tile_weights=None):
    """
    candidate_tile_descs: dict (tile_key -> desc)
    tile_weights: dict(tile_key->weight) default 1.0
    returns list of top_k tile keys sorted by weighted score
    """
    orb = cv2.ORB_create(ORB_MAX_FEATURES)
    qgray = cv2.cvtColor(query_frame, cv2.COLOR_RGB2GRAY)
    kq, dq = orb.detectAndCompute(qgray, None)
    if dq is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    scores = []
    for tile_key, desc in candidate_tile_descs.items():
        if desc is None:
            scores.append((tile_key, 0.0))
            continue
        try:
            matches = bf.match(dq, desc)
            score = float(len(matches))
            weight = tile_weights.get(tile_key, 1.0) if tile_weights else 1.0
            scores.append((tile_key, score * weight))
        except Exception:
            scores.append((tile_key, 0.0))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]

# ---------------- LoFTR matcher ----------------
class Localizer:
    def __init__(self, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[LoFTR] init on {self.device}")
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
                inliers = int(inlier_mask.sum()) if inlier_mask is not None else None
                return float(dst_pt[0][0][0]), float(dst_pt[0][0][1]), {'inliers': inliers, 'matches': len(mk0)}
        return None, None, None

# ---------------- simple motion prior ----------------
class MotionPrior:
    def __init__(self):
        self.xt = None; self.yt = None; self.vx = 0.0; self.vy = 0.0; self.last_ts = None
    def update_position(self, xt, yt, ts=None):
        now = ts if ts is not None else time.time()
        if self.xt is None:
            self.xt, self.yt, self.last_ts = xt, yt, now; return
        dt = now - self.last_ts if self.last_ts is not None else 1e-3
        self.vx = (xt - self.xt) / dt if dt>0 else 0.0
        self.vy = (yt - self.yt) / dt if dt>0 else 0.0
        self.xt, self.yt, self.last_ts = xt, yt, now
    def predict(self, dt=0.1):
        if self.xt is None: return None, None
        return self.xt + self.vx*dt, self.yt + self.vy*dt

# ---------------- route helpers ----------------
def load_route_file(route_path):
    if not os.path.exists(route_path): raise FileNotFoundError(route_path)
    if route_path.lower().endswith(".json"):
        with open(route_path, "r") as f:
            data = json.load(f)
        pts = data.get("waypoints") or data.get("route") or data.get("points")
        if pts is None: raise ValueError("route JSON must contain 'waypoints'")
        out = []
        for a in pts:
            if isinstance(a, (list,tuple)):
                out.append((float(a[0]), float(a[1])))
            else:
                out.append((float(a['lat']), float(a['lon'])))
        return out
    else:
        txt = open(route_path, "r").read()
        import re
        pts = []
        for m in re.finditer(r'<trkpt\\s+lat=\"([0-9\\.-]+)\"\\s+lon=\"([0-9\\.-]+)\"', txt):
            pts.append((float(m.group(1)), float(m.group(2))))
        if not pts: raise ValueError("GPX parse found 0 points")
        return pts

def waypoints_to_tiles(waypoints, zoom):
    tiles = set()
    for lat, lon in waypoints:
        xt, yt = deg2num(lat, lon, zoom)
        tiles.add((xt, yt))
    return sorted(list(tiles))

def tile_neighborhood(tile_set, radius=1):
    nbrs = set()
    for xt, yt in tile_set:
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nbrs.add((xt+dx, yt+dy))
    return sorted(list(nbrs))

# ---------------- draw map with trail ----------------
def render_map_with_trail(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, trail_stitched, out_display_h):
    # map_img is PIL RGB. We'll scale it to have height = out_display_h, preserving aspect.
    w,h = map_img.size
    scale = out_display_h / float(h)
    display_w = int(w * scale)
    display_h = out_display_h
    img_disp = map_img.resize((display_w, display_h)).copy()
    draw = ImageDraw.Draw(img_disp)
    # draw candidate tile boxes (scaled)
    for xt,yt in candidate_tiles:
        x0,y0,x1,y1 = tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile)
        x0 *= scale; y0 *= scale; x1 *= scale; y1 *= scale
        draw.rectangle((x0,y0,x1,y1), outline=(255,153,0), width=2)
    # draw route tiles as small green dots and lines
    centers = []
    for xt,yt in route_tiles:
        cx = ((xt - tl_xtile) * TILE_SIZE + TILE_SIZE/2) * scale
        cy = ((yt - tl_ytile) * TILE_SIZE + TILE_SIZE/2) * scale
        centers.append((cx, cy))
        r=4; draw.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(0,255,0))
    if len(centers)>1:
        draw.line(centers, fill=(0,255,0), width=2)
    # draw trail (list of stitched coords)
    if trail_stitched:
        pts = []
        for sx,sy in trail_stitched:
            pts.append(((sx * scale), (sy * scale)))
        if len(pts)>1:
            draw.line(pts, fill=(255,0,0), width=3)
        # latest point marker
        lx,ly = pts[-1]
        draw.ellipse((lx-6,ly-6,lx+6,ly+6), fill=(255,0,0))
    return img_disp

# ---------------- main pipeline ----------------
def run(
    video_path, route_path, out_dir,
    zoom=DEFAULT_ZOOM, grid_size=DEFAULT_GRID,
    search_radius=1, match_interval=MATCH_INTERVAL, top_k=TOP_K_TILES
):
    os.makedirs(out_dir, exist_ok=True)
    waypoints = load_route_file(route_path)
    route_tiles = waypoints_to_tiles(waypoints, zoom)
    if not route_tiles:
        raise RuntimeError("route generated no tiles")
    # center map
    mean_lat = sum([p[0] for p in waypoints]) / len(waypoints)
    mean_lon = sum([p[1] for p in waypoints]) / len(waypoints)
    map_img, tl_xtile, tl_ytile = download_and_stitch_map(mean_lat, mean_lon, zoom, grid_size, out_dir)
    map_w, map_h = map_img.size
    # candidate tiles
    candidate_tiles = tile_neighborhood(route_tiles, radius=search_radius)
    tile_descs = precompute_tile_descriptors_for_set(map_img, candidate_tiles, tl_xtile, tl_ytile)
    # localizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    localizer = Localizer(device=device)
    motion = MotionPrior()
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError("Failed to open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    # composite video writer: left=video (vw x vh), right=map (map_disp_w x vh) -> combined width
    # We'll compute map display width by scaling map_img to height=vh
    map_scale = vh / float(map_h)
    map_disp_w = int(map_w * map_scale)
    out_w = vw + map_disp_w
    out_h = vh
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = os.path.join(out_dir, "output_annotated.mp4")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, out_h))
    frame_idx = 0
    trail = []  # list of stitched (x,y)
    trajectory = []
    progressive_fail = 0
    sigma_route = max(1.0, len(route_tiles)/6.0)  # used for weighting
    print(f"[RUN] video {video_path} frames={total_frames} fps={fps} size={vw}x{vh}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        matched_flag = False
        latlon = (None, None)
        inliers = None
        # compute expected_index along route (progress fraction)
        progress_frac = float(frame_idx) / float(max(1, total_frames))
        expected_index = progress_frac * max(0, len(route_tiles)-1)
        # prepare tile_weights based on proximity along route to expected_index
        tile_weights = {}
        # compute route index mapping (for each candidate tile find nearest route index)
        for t in candidate_tiles:
            # nearest route tile index by Manhattan distance in tile-space
            nearest_idx = min(range(len(route_tiles)), key=lambda i: abs(route_tiles[i][0]-t[0])+abs(route_tiles[i][1]-t[1]))
            # weight: Gaussian-like around expected_index
            w = math.exp(-((nearest_idx - expected_index)**2) / (2.0 * sigma_route * sigma_route))
            # boost early frames toward start: additional temporal bias
            # If progress_frac is small (<0.2), multiply weight for early route indices
            if progress_frac < 0.25:
                # emphasize start of route
                start_bias = math.exp(-((nearest_idx - 0)**2) / (2.0 * (sigma_route*0.8)**2))
                w *= (1.0 + 1.2 * start_bias * (0.25 - progress_frac))  # stronger when earlier in video
            tile_weights[t] = w
        # run matching at interval
        if (frame_idx % match_interval) == 0:
            # ensure descriptors present
            missing = [t for t in candidate_tiles if t not in tile_descs]
            if missing:
                tile_descs.update(precompute_tile_descriptors_for_set(map_img, missing, tl_xtile, tl_ytile))
            top_tiles = find_best_tiles({k: tile_descs[k] for k in candidate_tiles}, frame_rgb, top_k=top_k, tile_weights=tile_weights)
            if len(top_tiles) == 0:
                progressive_fail += 1
                trajectory.append({"frame": frame_idx, "matched": False})
                if progressive_fail > 8:
                    search_radius = min(5, search_radius+1)
                    candidate_tiles = tile_neighborhood(route_tiles, radius=search_radius)
                    tile_descs = precompute_tile_descriptors_for_set(map_img, candidate_tiles, tl_xtile, tl_ytile)
                    progressive_fail = 0
            else:
                x0,y0,x1,y1 = union_bbox_from_tiles(top_tiles, tl_xtile, tl_ytile, TILE_SIZE, padding=PADDING_CROP, map_w=map_w, map_h=map_h)
                ref_crop = map_img.crop((x0,y0,x1,y1)).convert("RGB")
                ref_np = np.array(ref_crop)
                mx_rel, my_rel, info = localizer.match(ref_np, frame_rgb)
                if mx_rel is not None:
                    stitched_x = x0 + mx_rel
                    stitched_y = y0 + my_rel
                    # convert to lat lon
                    real_x = stitched_x / TILE_SIZE
                    real_y = stitched_y / TILE_SIZE
                    lat, lon = num2deg(tl_xtile + real_x, tl_ytile + real_y, zoom)
                    latlon = (lat, lon)
                    inliers = info.get("inliers") if info else None
                    # update motion prior and trail
                    tx = tl_xtile + int(stitched_x // TILE_SIZE)
                    ty = tl_ytile + int(stitched_y // TILE_SIZE)
                    motion.update_position(tx, ty)
                    trail.append((stitched_x, stitched_y))
                    if len(trail) > TRAIL_MAX_POINTS:
                        trail = trail[-TRAIL_MAX_POINTS:]
                    matched_flag = True
                    trajectory.append({"frame": frame_idx, "matched": True, "lat": lat, "lon": lon, "inliers": inliers})
                    progressive_fail = 0
                else:
                    progressive_fail += 1
                    trajectory.append({"frame": frame_idx, "matched": False, "reason": "loftr_fail"})
        else:
            # use motion prior to predict if available
            p = motion.predict(dt=0.1)
            if p[0] is not None:
                trajectory.append({"frame": frame_idx, "matched": False, "predicted_tile": p})
            else:
                trajectory.append({"frame": frame_idx, "matched": False})
        # compose output frame: left = camera frame, right = map display with trail (display height = vh)
        map_disp = render_map_with_trail(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, trail, out_display_h=vh)
        # convert map_disp (PIL) to BGR numpy
        map_np = cv2.cvtColor(np.array(map_disp), cv2.COLOR_RGB2BGR)
        # ensure left frame is size vw x vh
        left_frame = cv2.resize(frame, (vw, vh))
        # combine horizontally
        combined = np.zeros((vh, vw + map_np.shape[1], 3), dtype=np.uint8)
        combined[:, :vw, :] = left_frame
        combined[:, vw:, :] = map_np
        # overlay small text telemetry
        if matched_flag:
            text = f"LAT: {latlon[0]:.6f} LON: {latlon[1]:.6f}"
        else:
            text = "GPS: STANDBY"
        cv2.rectangle(combined, (8,8), (600,56), (0,0,0), thickness=-1)
        cv2.putText(combined, text, (16,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        writer.write(combined)
    # finish
    cap.release(); writer.release()
    # save outputs
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(trajectory, f, indent=2)
    # save debug map with full trail
    debug_map_path = os.path.join(out_dir, "debug_map.png")
    render_map_with_trail(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, trail, out_display_h=map_h).save(debug_map_path)
    print("[DONE] outputs in", out_dir)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--outdir", default="./vns_out")
    p.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    p.add_argument("--grid", type=int, default=DEFAULT_GRID)
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--match_interval", type=int, default=MATCH_INTERVAL)
    p.add_argument("--topk", type=int, default=TOP_K_TILES)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.video, args.route, args.outdir, zoom=args.zoom, grid_size=args.grid, search_radius=args.radius, match_interval=args.match_interval, top_k=args.topk)
    