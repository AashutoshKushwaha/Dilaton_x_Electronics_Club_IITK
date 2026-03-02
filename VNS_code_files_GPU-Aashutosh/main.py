#!/usr/bin/env python3
"""
main_advanced_vns.py

Combined: batch pipeline + interactive Tk GUI for live visual navigation.

Usage (batch):
  python main_advanced_vns.py --video <video> --route <route.json> --outdir <outdir> [--use_netvlad ...]

Usage (GUI):
  python main_advanced_vns.py --gui --video <video> --map <stitched_map.png> \
       --descs <descriptors.npy> --imgpaths <image_paths.txt> [--tile_size 256] [--force_cpu]

Notes:
 - Set ARCGIS_API_KEY environment variable for map download in batch mode.
 - GUI expects:
     * stitched map image (full_map)
     * precomputed global descriptors (descriptors.npy)
     * corresponding image paths list (one per descriptor) in same order (image_paths.txt)
 - The GUI uses LoFTR matcher (Kornia or EfficientLoFTR if available).
"""

import os
import sys
import math
import json
import time
import argparse
import io
import warnings
import re
from typing import List, Tuple

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import cv2
import torch

# Try import tkinter lazily if GUI requested
try:
    import tkinter as tk
except Exception:
    tk = None

# ---------- Optional libs detection ----------
HAS_FAISS = False
HAS_NETVLAD = False
HAS_EFFICIENTLOFTR = False
HAS_KORNIA_LOFTR = False

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import hloc.extractors.netvlad as _netvlad_mod  # type: ignore
    HAS_NETVLAD = True
except Exception:
    HAS_NETVLAD = False

try:
    from efficientloftr import EfficientLoFTR  # type: ignore
    HAS_EFFICIENTLOFTR = True
except Exception:
    HAS_EFFICIENTLOFTR = False

try:
    from kornia.feature import LoFTR  # type: ignore
    HAS_KORNIA_LOFTR = True
except Exception:
    HAS_KORNIA_LOFTR = False

# -------------------------
# Defaults and constants
# -------------------------
DEFAULT_ZOOM = 18
DEFAULT_GRID = 5
TILE_SIZE = 256
MATCH_INTERVAL = 6
TOP_K_TILES = 2
ORB_MAX_FEATURES = 800
MIN_LOFTR_MATCHES = 12
PADDING_CROP = 96
TRAIL_MAX_POINTS = 300

# -------------------------
# Geo / Tile helpers
# -------------------------
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

# -------------------------
# Map download and cache (batch)
# -------------------------
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
    headers = {'User-Agent': 'VNS-Advanced/1.0'}
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

# -------------------------
# Retrieval indexes
# -------------------------
class ORBRetrievalIndex:
    """Fallback retrieval: compute ORB descriptors per tile and rank by match count."""
    def __init__(self, map_img, tile_list, tl_xtile, tl_ytile, max_features=ORB_MAX_FEATURES):
        self.orb = cv2.ORB_create(max_features)
        self.tile_descs = {}
        self.tile_list = list(tile_list)
        self.map_w, self.map_h = map_img.size
        for xt,yt in self.tile_list:
            x0,y0,x1,y1 = tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile)
            x0 = max(0, min(self.map_w-1, x0)); y0 = max(0, min(self.map_h-1, y0))
            x1 = max(0, min(self.map_w, x1)); y1 = max(0, min(self.map_h, y1))
            tile = map_img.crop((x0,y0,x1,y1)).convert("RGB")
            tnp = np.array(tile)
            gray = cv2.cvtColor(tnp, cv2.COLOR_RGB2GRAY)
            k,d = self.orb.detectAndCompute(gray, None)
            self.tile_descs[(xt,yt)] = d

    def query(self, frame_rgb, topk=TOP_K_TILES, tile_weights=None):
        qgray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        kq, dq = self.orb.detectAndCompute(qgray, None)
        if dq is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        scores = []
        for t, desc in self.tile_descs.items():
            if desc is None:
                scores.append((t, 0.0))
                continue
            try:
                matches = bf.match(dq, desc)
                score = float(len(matches))
            except Exception:
                score = 0.0
            w = tile_weights.get(t, 1.0) if tile_weights else 1.0
            scores.append((t, score * w))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:topk]]

class NetVLADFaissIndex:
    """
    NetVLAD + FAISS index wrapper.
    Expects a dict tile_key -> vector (1D float32 numpy).
    """
    def __init__(self, tile_descs_dict=None):
        if not HAS_FAISS:
            raise RuntimeError("FAISS is required for NetVLAD retrieval (pip install faiss-cpu or faiss-gpu)")
        if tile_descs_dict is None or len(tile_descs_dict)==0:
            raise RuntimeError("NetVLADFaissIndex requires precomputed tile descriptors (tile->vec).")
        self.tile_keys = list(tile_descs_dict.keys())
        vectors = np.stack([tile_descs_dict[k] for k in self.tile_keys]).astype('float32')
        d = vectors.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(vectors)
        self.vectors = vectors

    def query(self, qvec, topk=10):
        q = qvec.astype('float32')[None, :]
        D, I = self.index.search(q, topk)
        results = [(self.tile_keys[int(i)], float(D[0, j])) for j, i in enumerate(I[0])]
        return results

# -------------------------
# Matchers (LoFTR/EfficientLoFTR wrapper)
# -------------------------
class LocalizerLoFTR:
    """Wrapper that uses EfficientLoFTR when available, else Kornia.LoFTR"""
    def __init__(self, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = None
        self.matcher = None
        if HAS_EFFICIENTLOFTR:
            try:
                self.matcher = EfficientLoFTR(pretrained=True).eval().to(self.device)
                self.mode = 'efficient'
                print("[MATCHER] Using EfficientLoFTR")
            except Exception as e:
                print("[MATCHER] EfficientLoFTR available but init failed:", e)
                self.matcher = None
                self.mode = None
        if self.mode is None:
            if HAS_KORNIA_LOFTR:
                self.matcher = LoFTR(pretrained='outdoor').eval().to(self.device)
                self.mode = 'kornia'
                print("[MATCHER] Using Kornia LoFTR (fallback)")
            else:
                raise RuntimeError("No LoFTR implementation available. Install kornia or efficientloftr.")

    def match(self, ref_rgb_np, query_rgb_np, min_matches=MIN_LOFTR_MATCHES):
        try:
            gray_ref = cv2.cvtColor(ref_rgb_np, cv2.COLOR_RGB2GRAY)
            gray_query = cv2.cvtColor(query_rgb_np, cv2.COLOR_RGB2GRAY)
        except Exception:
            return None, None, None

        t_ref = torch.from_numpy(gray_ref).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
        t_query = torch.from_numpy(gray_query).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            corresp = self.matcher({'image0': t_query, 'image1': t_ref})

        mkpts0 = corresp.get('keypoints0')
        mkpts1 = corresp.get('keypoints1')
        if mkpts0 is None or mkpts1 is None:
            return None, None, None
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()

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
                hq, wq = gray_query.shape
                center = np.array([[[wq/2, hq/2]]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(center, H)
                inliers = int(inlier_mask.sum()) if inlier_mask is not None else None
                info = {'inliers': inliers, 'matches': len(mk0)}
                return float(dst_pt[0][0][0]), float(dst_pt[0][0][1]), info
        return None, None, None

# -------------------------
# Kalman filter + route projection helpers
# -------------------------
class KalmanFilter2D:
    def __init__(self, init_x=0.0, init_y=0.0, dt=1/20.0, process_var=1.0, meas_var=25.0):
        self.dt = float(dt)
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 1e3
        self.F = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        q = float(process_var)
        self.Q = np.array([
            [q*(self.dt**4)/4, 0, q*(self.dt**3)/2, 0],
            [0, q*(self.dt**4)/4, 0, q*(self.dt**3)/2],
            [q*(self.dt**3)/2, 0, q*(self.dt**2), 0],
            [0, q*(self.dt**3)/2, 0, q*(self.dt**2)]
        ], dtype=float)
        self.R = np.eye(2) * float(meas_var)

    def predict(self, dt=None):
        if dt is not None and dt > 0:
            self.dt = float(dt)
            self.F = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x.copy()

    def update(self, meas_xy, meas_var=None):
        z = np.array([meas_xy[0], meas_xy[1]], dtype=float)
        if meas_var is not None:
            R = np.eye(2) * float(meas_var)
        else:
            R = self.R
        S = self.H.dot(self.P).dot(self.H.T) + R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        I = np.eye(4)
        self.P = (I - K.dot(self.H)).dot(self.P)
        return self.x.copy()
    def state(self):
        return self.x.copy()
def project_point_to_route(pt: Tuple[float,float], route_pts: List[Tuple[float,float]]):
    px, py = float(pt[0]), float(pt[1])
    best_d2 = float("inf")
    best_proj = (px, py)
    best_idx = 0
    best_t = 0.0
    for i in range(len(route_pts)-1):
        x1, y1 = route_pts[i]
        x2, y2 = route_pts[i+1]
        vx, vy = x2 - x1, y2 - y1
        denom = vx*vx + vy*vy
        if denom == 0:
            t = 0.0
            projx, projy = x1, y1
        else:
            t = ((px - x1)*vx + (py - y1)*vy) / denom
            if t < 0: t = 0.0
            elif t > 1: t = 1.0
            projx = x1 + t*vx
            projy = y1 + t*vy
        d2 = (projx - px)**2 + (projy - py)**2
        if d2 < best_d2:
            best_d2 = d2
            best_proj = (projx, projy)
            best_idx = i
            best_t = t
    return best_proj[0], best_proj[1], best_idx, best_t, best_d2

def route_tiles_to_pixel_centers(route_tiles, tl_xtile, tl_ytile, tile_size=TILE_SIZE):
    pts = []
    for xt, yt in route_tiles:
        cx = ((xt - tl_xtile) * tile_size) + tile_size * 0.5
        cy = ((yt - tl_ytile) * tile_size) + tile_size * 0.5
        pts.append((cx, cy))
    return pts

# -------------------------
# Rendering helpers
# -------------------------
def render_map_with_trail(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, trail_stitched, out_display_h):
    w,h = map_img.size
    scale = out_display_h / float(h)
    display_w = int(w * scale)
    display_h = out_display_h
    img_disp = map_img.resize((display_w, display_h)).copy()
    draw = ImageDraw.Draw(img_disp)
    for xt,yt in candidate_tiles:
        x0,y0,x1,y1 = tile_to_pixel_bbox(xt, yt, tl_xtile, tl_ytile)
        x0 *= scale; y0 *= scale; x1 *= scale; y1 *= scale
        draw.rectangle((x0,y0,x1,y1), outline=(255,153,0), width=2)
    centers = []
    for xt,yt in route_tiles:
        cx = ((xt - tl_xtile) * TILE_SIZE + TILE_SIZE/2) * scale
        cy = ((yt - tl_ytile) * TILE_SIZE + TILE_SIZE/2) * scale
        centers.append((cx, cy))
        r=3; draw.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(0,255,0))
    if len(centers)>1:
        draw.line(centers, fill=(0,255,0), width=2)
    if trail_stitched:
        pts = [((sx * scale), (sy * scale)) for sx,sy in trail_stitched]
        if len(pts)>1:
            draw.line(pts, fill=(0,0,255), width=4)
        lx,ly = pts[-1]
        draw.ellipse((lx-6,ly-6,lx+6,ly+6), fill=(0,0,255))
    return img_disp

# -------------------------
# Route loader & helpers
# -------------------------
def load_route_file(route_path):
    if not os.path.exists(route_path):
        raise FileNotFoundError(route_path)
    if route_path.lower().endswith(".json"):
        with open(route_path, "r") as f:
            data = json.load(f)
        w = data.get("waypoints") or data.get("route") or data.get("points")
        if w is None:
            raise ValueError("route JSON must contain 'waypoints' (list of [lat,lon])")
        out = []
        for a in w:
            if isinstance(a, (list,tuple)):
                out.append((float(a[0]), float(a[1])))
            else:
                out.append((float(a['lat']), float(a['lon'])))
        return out
    else:
        txt = open(route_path, "r").read()
        pts = []
        for m in re.finditer(r'<trkpt\s+lat=\"([0-9\.\-]+)\"\s+lon=\"([0-9\.\-]+)\"', txt):
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

# -------------------------
# Pipeline runner (batch)
# -------------------------
def run_pipeline(
    video_path, route_path, out_dir,
    zoom=DEFAULT_ZOOM, grid_size=DEFAULT_GRID,
    search_radius=1, match_interval=MATCH_INTERVAL, top_k=TOP_K_TILES,
    use_netvlad=False
):
    os.makedirs(out_dir, exist_ok=True)

    waypoints = load_route_file(route_path)
    route_tiles = waypoints_to_tiles(waypoints, zoom)
    mean_lat = sum([p[0] for p in waypoints]) / len(waypoints)
    mean_lon = sum([p[1] for p in waypoints]) / len(waypoints)
    map_img, tl_xtile, tl_ytile = download_and_stitch_map(mean_lat, mean_lon, zoom, grid_size, out_dir)
    map_w, map_h = map_img.size

    candidate_tiles = tile_neighborhood(route_tiles, radius=search_radius)

    retrieval_index = None
    if use_netvlad:
        if not (HAS_FAISS):
            warnings.warn("FAISS not available -> falling back to ORB retrieval.")
            use_netvlad = False

    if use_netvlad:
        nv_path = os.path.join(out_dir, "netvlad_tile_descs.npz")
        if os.path.exists(nv_path):
            data = np.load(nv_path, allow_pickle=True)
            tile_keys = data["tile_keys"].tolist()
            vectors = data["vectors"]
            netvlad_tile_descs = {tuple(k): vectors[i] for i,k in enumerate(tile_keys)}
            retrieval_index = NetVLADFaissIndex(netvlad_tile_descs)
            print("[NETVLAD] loaded NetVLAD descriptors + faiss index")
        else:
            warnings.warn(f"NetVLAD selected but '{nv_path}' not found. Falling back to ORB retrieval.")
            use_netvlad = False

    if not use_netvlad:
        print("[RETRIEVAL] Using ORB retrieval (fallback).")
        retrieval_index = ORBRetrievalIndex(map_img, candidate_tiles, tl_xtile, tl_ytile)

    localizer = LocalizerLoFTR()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: " + str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    map_scale = vh / float(map_h)
    map_disp_w = int(map_w * map_scale)
    out_w = vw + map_disp_w
    out_h = vh
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(out_dir, "output_annotated.mp4")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))

    route_pixel_centers = route_tiles_to_pixel_centers(route_tiles, tl_xtile, tl_ytile, tile_size=TILE_SIZE)
    kf = None
    MEAS_VAR_BASE = 100.0
    PROCESS_VAR = 1.0
    MAX_MEAS_DIST = 800.0
    last_update_ts = None
    frame_time_base = 1.0 / max(1.0, fps)

    trail = []
    trajectory = []
    progressive_fail = 0
    sigma_route = max(1.0, len(route_tiles)/6.0)
    frame_idx = 0
    last_latlon = (None, None)

    def encode_frame_for_netvlad(frame_rgb):
        try:
            from netvlad_utils import netvlad_encode  # user-supplied
            return netvlad_encode(frame_rgb)
        except Exception as e:
            raise RuntimeError("NetVLAD chosen but 'netvlad_utils.netvlad_encode' not available.") from e

    print("[RUN] Starting pipeline...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        matched_flag = False
        latlon = (None, None)
        inliers = None

        progress_frac = float(frame_idx) / float(max(1, total_frames))
        expected_index = progress_frac * max(0, len(route_tiles)-1)

        tile_weights = {}
        for t in candidate_tiles:
            nearest_idx = min(range(len(route_tiles)), key=lambda i: abs(route_tiles[i][0]-t[0])+abs(route_tiles[i][1]-t[1]))
            w = math.exp(-((nearest_idx - expected_index)**2) / (2.0 * sigma_route * sigma_route))
            if progress_frac < 0.25:
                start_bias = math.exp(-((nearest_idx - 0)**2) / (2.0 * (sigma_route*0.8)**2))
                w *= (1.0 + 1.2 * start_bias * (0.25 - progress_frac))
            tile_weights[t] = w

        if use_netvlad and isinstance(retrieval_index, NetVLADFaissIndex):
            qvec = encode_frame_for_netvlad(frame_rgb)
            net_results = retrieval_index.query(qvec, topk=20)
            candidates_sorted = [r[0] for r in net_results]
            candidates_sorted = sorted(candidates_sorted, key=lambda t: -tile_weights.get(t,1.0))[:top_k]
        else:
            candidates_sorted = retrieval_index.query(frame_rgb, topk=top_k, tile_weights=tile_weights)

        mx_rel = None; my_rel = None; info = None
        if (frame_idx % match_interval) == 0 and len(candidates_sorted) > 0:
            x0,y0,x1,y1 = union_bbox_from_tiles(candidates_sorted, tl_xtile, tl_ytile, TILE_SIZE, padding=PADDING_CROP, map_w=map_w, map_h=map_h)
            ref_crop = map_img.crop((x0,y0,x1,y1)).convert("RGB")
            ref_np = np.array(ref_crop)
            mx_rel, my_rel, info = localizer.match(ref_np, frame_rgb)

        if mx_rel is not None:
            stitched_x = x0 + mx_rel
            stitched_y = y0 + my_rel
            proj_x, proj_y, seg_idx, along_t, d2 = project_point_to_route((stitched_x, stitched_y), route_pixel_centers)
            meas_dist = math.sqrt(d2)
            real_x = stitched_x / TILE_SIZE
            real_y = stitched_y / TILE_SIZE
            lat, lon = num2deg(tl_xtile + real_x, tl_ytile + real_y, zoom)
            latlon = (lat, lon)
            inliers = info.get("inliers") if info else None
            route_prox = max([tile_weights.get(t,1.0) for t in candidates_sorted]) if candidates_sorted else 1.0
            loftr_conf = (inliers or 0) / 50.0
            conf = 0.6 * loftr_conf + 0.3 * route_prox

            now_ts = time.time()
            dt = frame_time_base if last_update_ts is None else max(1e-3, now_ts - last_update_ts)
            last_update_ts = now_ts

            if kf is None:
                kf = KalmanFilter2D(init_x=proj_x, init_y=proj_y, dt=dt, process_var=PROCESS_VAR, meas_var=MEAS_VAR_BASE)
                filtered_state = kf.state()
                matched_flag = True
                trail.append((proj_x, proj_y))
                trajectory.append({"frame": frame_idx, "matched": True, "lat": lat, "lon": lon, "inliers": inliers, "conf": conf, "filtered": (proj_x, proj_y)})
                last_latlon = latlon
                progressive_fail = 0
            else:
                pred = kf.predict(dt=dt)
                pred_xy = (pred[0], pred[1])
                pred_error = math.hypot(proj_x - pred_xy[0], proj_y - pred_xy[1])
                meas_var = MEAS_VAR_BASE * (1.0 + max(0.0, (50.0 - (inliers or 0))/50.0))
                if pred_error < MAX_MEAS_DIST:
                    R_adj = meas_var * max(0.25, 1.0 - conf)
                    kf.R = np.eye(2) * R_adj
                    st = kf.update((proj_x, proj_y), meas_var=R_adj)
                    filtered_xy = (st[0], st[1])
                    matched_flag = True
                    trail.append((filtered_xy[0], filtered_xy[1]))
                    if len(trail) > TRAIL_MAX_POINTS:
                        trail = trail[-TRAIL_MAX_POINTS:]
                    trajectory.append({"frame": frame_idx, "matched": True, "lat": lat, "lon": lon, "inliers": inliers, "conf": conf, "filtered": filtered_xy})
                    last_latlon = latlon
                    progressive_fail = 0
                else:
                    pred2 = kf.predict(dt=dt)
                    filtered_xy = (pred2[0], pred2[1])
                    matched_flag = False
                    progressive_fail += 1
                    trajectory.append({"frame": frame_idx, "matched": False, "reason": "outlier_pred", "pred_error": pred_error})
        else:
            if kf is not None:
                pred = kf.predict(dt=frame_time_base)
                filtered_xy = (pred[0], pred[1])
                trajectory.append({"frame": frame_idx, "matched": False, "predicted_tile": (pred[0], pred[1])})
            else:
                trajectory.append({"frame": frame_idx, "matched": False})

        map_disp = render_map_with_trail(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, trail, out_display_h=vh)
        map_np = cv2.cvtColor(np.array(map_disp), cv2.COLOR_RGB2BGR)
        left_frame = cv2.resize(frame, (vw, vh))
        combined = np.zeros((vh, vw + map_np.shape[1], 3), dtype=np.uint8)
        combined[:, :vw, :] = left_frame
        combined[:, vw:, :] = map_np
        overlay_txt = f"LAT: {last_latlon[0]:.6f} LON: {last_latlon[1]:.6f}" if last_latlon[0] is not None else "GPS: STANDBY"
        cv2.rectangle(combined, (8,8), (600,56), (0,0,0), thickness=-1)
        cv2.putText(combined, overlay_txt, (16,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        writer.write(combined)

    cap.release()
    writer.release()
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(trajectory, f, indent=2)
    debug_map_path = os.path.join(out_dir, "debug_map.png")
    render_map_with_trail(map_img, tl_xtile, tl_ytile, route_tiles, candidate_tiles, trail, out_display_h=map_h).save(debug_map_path)
    print("[DONE] outputs written to", out_dir)

# -------------------------
# GUI app (Tkinter) - live viewer
# -------------------------
class VNSApp:
    def __init__(self, root, video_path, stitched_map_path, descs_path, imgpaths_file, tile_size=256, match_every_n_frames=30, top_k=2, match_w=640, match_h=360, force_cpu=False):
        if tk is None:
            raise RuntimeError("tkinter is not available on this Python build.")
        self.root = root
        self.root.title("VNS – Live Visual Navigation")
        self.match_every_n_frames = int(match_every_n_frames)
        self.top_k = int(top_k)
        self.match_w = int(match_w); self.match_h = int(match_h)
        self.TILE_SIZE = int(tile_size)
        self.force_cpu = bool(force_cpu)
        # Device
        self.device = "cpu" if self.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        # Load precomputed descriptors & image paths
        if not os.path.exists(descs_path):
            raise FileNotFoundError(descs_path)
        if not os.path.exists(imgpaths_file):
            raise FileNotFoundError(imgpaths_file)
        self.map_descs = np.load(descs_path)  # shape (N, D)
        with open(imgpaths_file, "r") as f:
            self.map_img_paths = [l.strip() for l in f.readlines()]
        # LoFTR localizer
        self.localizer = LocalizerLoFTR(device=self.device)
        # Video & map
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video: " + str(video_path))
        full_map = cv2.imread(stitched_map_path)
        if full_map is None:
            raise RuntimeError("Failed to open stitched map image: " + str(stitched_map_path))
        self.full_map = full_map.copy()
        # UI layout
        WINDOW_W, WINDOW_H = 920, 520
        PANEL_SIZE = 480
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = max(0, (screen_w - WINDOW_W) // 2)
        y = max(0, (screen_h - WINDOW_H) // 2)
        root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")
        root.resizable(True, True)
        main = tk.Frame(root)
        main.pack(padx=6, pady=6, fill=tk.BOTH, expand=True)
        # Map panel
        self.map_label = tk.Label(main)
        self.map_label.grid(row=0, column=0, padx=6, pady=6)
        # Video panel
        self.video_label = tk.Label(main)
        self.video_label.grid(row=0, column=1, padx=6, pady=6)
        # Control button
        self.tracking = False
        self.frame_id = 0
        self.current_location = None
        self.btn = tk.Button(root, text="START TRACKING", font=("Arial", 12, "bold"), bg="green", fg="white", width=20, command=self.toggle_tracking)
        self.btn.pack(pady=6)
        self.update_loop()

    def toggle_tracking(self):
        self.tracking = not self.tracking
        if self.tracking:
            self.btn.config(text="STOP TRACKING", bg="red")
        else:
            self.btn.config(text="START TRACKING", bg="green")

    def compute_global_descriptor(self, img_bgr):
        # simple handcrafted descriptor (HSV hist + edge histogram) used for retrieval
        img = cv2.resize(img_bgr, (128, 128))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h1 = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        h2 = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        edges = cv2.Canny(img, 100, 200)
        h3, _ = np.histogram(edges, bins=16, range=(0, 256))
        desc = np.concatenate([h1, h2, h3]).astype(np.float32)
        norm = np.linalg.norm(desc) + 1e-6
        return desc / norm

    def parse_tile_rc_from_path(self, path):
        basename = os.path.basename(path)
        # find pairs of integers in basename; take last two as (r,c) or (x,y)
        nums = re.findall(r'(\d+)', basename)
        if len(nums) >= 2:
            # return integers as (r,c)
            try:
                a = int(nums[-2]); b = int(nums[-1])
                return a, b
            except:
                return None
        return None

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        self.frame_id += 1

        if self.tracking and (self.frame_id % self.match_every_n_frames == 0):
            frame_small = cv2.resize(frame, (self.match_w, self.match_h))
            drone_desc = self.compute_global_descriptor(frame_small)
            # dot product similarity
            scores = self.map_descs @ drone_desc
            top_ids = np.argsort(scores)[-self.top_k:][::-1]

            best_score = -1
            best_H = None
            best_tile = None

            for idx in top_ids:
                tile_path = self.map_img_paths[idx]
                map_img = cv2.imread(tile_path)
                if map_img is None:
                    continue
                # Resize query or let LoFTR handle different sizes
                q_img = cv2.resize(frame_small, (map_img.shape[1], map_img.shape[0]))
                # LoFTR requires RGB uint8 arrays
                ref_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
                qry_rgb = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
                mx, my, info = self.localizer.match(ref_rgb, qry_rgb)
                if mx is None:
                    continue
                # We approximate number of matches by info['matches'] if available
                matches_count = info.get('matches') if info and 'matches' in info else 0
                if matches_count < 8 and matches_count < best_score:
                    continue
                # compute homography similar to earlier (we already used match -> homography inside localizer)
                # localizer returned projection of center into ref coords -> we can use that directly
                # We'll accept whichever has larger matches_count
                if matches_count > best_score:
                    best_score = matches_count
                    # compute tile pixel offset
                    best_H = None
                    best_tile = tile_path
                    # Use mx,my as relative within ref image
                    rel_x, rel_y = float(mx), float(my)
                    # compute stitched map pixel
                    rc = self.parse_tile_rc_from_path(tile_path)
                    if rc is not None:
                        r, c = rc
                        px = int(c * self.TILE_SIZE + rel_x)
                        py = int(r * self.TILE_SIZE + rel_y)
                        self.current_location = (px, py)
                    else:
                        # fallback: approximate by centering on map
                        h, w = self.full_map.shape[:2]
                        self.current_location = (w//2, h//2)

        # draw on stitched map
        map_vis = self.full_map.copy()
        if self.current_location is not None:
            cv2.circle(map_vis, self.current_location, 8, (0, 0, 255), -1)

        map_vis_disp = cv2.resize(map_vis, (480, 480))
        map_imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(map_vis_disp, cv2.COLOR_BGR2RGB)))
        self.map_label.configure(image=map_imgtk)
        self.map_label.image = map_imgtk

        video_vis = cv2.resize(frame, (480, 480))
        vid_imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(video_vis, cv2.COLOR_BGR2RGB)))
        self.video_label.configure(image=vid_imgtk)
        self.video_label.image = vid_imgtk

        self.root.after(30, self.update_loop)

# -------------------------
# CLI & entrypoint
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="Path to input video (required for both batch and GUI)", required=False)
    p.add_argument("--route", help="Route JSON/GPX file (required for batch)", required=False)
    p.add_argument("--outdir", help="Output directory for batch", default="./vns_out")
    p.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    p.add_argument("--grid", type=int, default=DEFAULT_GRID)
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--match_interval", type=int, default=MATCH_INTERVAL)
    p.add_argument("--topk", type=int, default=TOP_K_TILES)
    p.add_argument("--use_netvlad", action="store_true")
    # GUI flags
    p.add_argument("--gui", action="store_true", help="Run interactive Tk GUI (uses descriptors & image paths)")
    p.add_argument("--map", help="Stitched map image (for GUI)", required=False)
    p.add_argument("--descs", help="Descriptors .npy (for GUI)", required=False)
    p.add_argument("--imgpaths", help="Image paths txt (for GUI)", required=False)
    p.add_argument("--tile_size", type=int, default=TILE_SIZE)
    p.add_argument("--match_every_n_frames", type=int, default=30)
    p.add_argument("--topk_gui", type=int, default=2)
    p.add_argument("--match_w", type=int, default=640)
    p.add_argument("--match_h", type=int, default=360)
    p.add_argument("--force_cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.gui:
        # Validate GUI-required inputs
        if args.video is None or args.map is None or args.descs is None or args.imgpaths is None:
            print("GUI mode requires --video, --map, --descs, and --imgpaths")
            sys.exit(1)
        # Start Tkinter GUI
        root = tk.Tk()
        app = VNSApp(root,
                     video_path=args.video,
                     stitched_map_path=args.map,
                     descs_path=args.descs,
                     imgpaths_file=args.imgpaths,
                     tile_size=args.tile_size,
                     match_every_n_frames=args.match_every_n_frames,
                     top_k=args.topk_gui,
                     match_w=args.match_w,
                     match_h=args.match_h,
                     force_cpu=args.force_cpu)
        root.mainloop()
    else:
        # Batch run; require video + route
        if args.video is None or args.route is None:
            print("Batch mode requires --video and --route arguments. Use --gui for interactive mode.")
            sys.exit(1)
        run_pipeline(args.video, args.route, args.outdir, zoom=args.zoom, grid_size=args.grid, search_radius=args.radius, match_interval=args.match_interval, top_k=args.topk, use_netvlad=args.use_netvlad)
