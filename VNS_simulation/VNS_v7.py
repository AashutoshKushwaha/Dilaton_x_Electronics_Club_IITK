import os
import math
import cv2
import torch
import requests
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from io import BytesIO
from kornia.feature import LoFTR

# --- CONFIGURATION ---
MAPTILER_API_KEY = "PLEASE ADD YOUR API KEY HERE from https://www.maptiler.com/cloud/"  # <-- Replace with your MapTiler API Key
LAT = 40.76161416837142
LON = -73.97837912560394
ZOOM_LEVEL = 18
TILE_SIZE = 256
GRID_SIZE = 5
CACHE_FILENAME = "map_cache_v2.png"

# AI & VISUAL CONFIG
SCREEN_BOX_SIZE = 250
MARKER_DISTANCE_THRESHOLD = 20  # Reduced to 20 so points appear faster

# CYCLE COLORS for the markers (Red, Green, Cyan, Yellow)
MARKER_COLORS = ["#FF3333", "#33FF33", "#33FFFF", "#FFFF33"] 

# --- BACKEND: MAPPING & CV ---
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

def download_and_stitch_map(lat, lon, zoom, grid_size=3):
    center_xtile, center_ytile = deg2num(lat, lon, zoom)
    offset = grid_size // 2
    tl_xtile = center_xtile - offset
    tl_ytile = center_ytile - offset

    if os.path.exists(CACHE_FILENAME):
        print(f"Loading map from cache: {CACHE_FILENAME}")
        return Image.open(CACHE_FILENAME).convert("RGB"), tl_xtile, tl_ytile

    print(f"Downloading {grid_size}x{grid_size} map tiles at Zoom {zoom}...")
    
    total_width = grid_size * TILE_SIZE
    total_height = grid_size * TILE_SIZE
    stitched_img = Image.new("RGB", (total_width, total_height))
    
    for x in range(grid_size):
        for y in range(grid_size):
            xtile = tl_xtile + x
            ytile = tl_ytile + y
            url = f"https://api.maptiler.com/maps/satellite/256/{zoom}/{xtile}/{ytile}.jpg?key={MAPTILER_API_KEY}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    tile = Image.open(BytesIO(response.content)).convert("RGB")
                    stitched_img.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))
            except Exception as e:
                print(f"Error fetching tile: {e}")
                
    print(f"Map download complete. Saving to {CACHE_FILENAME}...")
    stitched_img.save(CACHE_FILENAME)
    return stitched_img, tl_xtile, tl_ytile

class Localizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading LoFTR on {self.device}...")
        self.matcher = LoFTR(pretrained='outdoor').eval().to(self.device)
        
    def match(self, img_ref, img_query):
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
        gray_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)
        
        t_ref = torch.from_numpy(gray_ref).float()[None, None].to(self.device) / 255.0
        t_query = torch.from_numpy(gray_query).float()[None, None].to(self.device) / 255.0

        with torch.no_grad():
            input_dict = {"image0": t_query, "image1": t_ref}
            correspondences = self.matcher(input_dict)
            
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        
        if len(mkpts0) > 8:
            H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
            if H is not None:
                h, w = gray_query.shape
                center_pt = np.array([[[w/2, h/2]]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(center_pt, H)
                return dst_pt[0][0]
        return None

# --- FRONTEND: TKINTER UI ---

class DroneSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VNS Simulator v5 (4-Point HUD History)")
        self.root.geometry("1400x850")
        
        # Load Custom Font for GPS Text
        try:
            self.marker_font = ImageFont.truetype("arial.ttf", 24)
            self.hud_font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            try:
                self.marker_font = ImageFont.truetype("DejaVuSans.ttf", 24)
                self.hud_font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except IOError:
                print("Warning: Custom font not found. Using default.")
                self.marker_font = ImageFont.load_default()
                self.hud_font = ImageFont.load_default()
        
        # 1. Data Setup
        self.map_high_res, self.tl_xtile, self.tl_ytile = download_and_stitch_map(LAT, LON, ZOOM_LEVEL, GRID_SIZE)
        self.map_w, self.map_h = self.map_high_res.size
        
        self.ref_scale = 0.5
        self.map_ref = self.map_high_res.resize(
            (int(self.map_w * self.ref_scale), int(self.map_h * self.ref_scale)), 
            Image.Resampling.LANCZOS
        )
        self.ref_cv = np.array(self.map_ref)
        
        # Drone State
        self.drone_x = self.map_w // 2
        self.drone_y = self.map_h // 2
        self.altitude = 50
        
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.vns_active = False 
        
        # Markers Logic
        self.gps_markers = [] 
        self.last_marker_pos = None
        self.marker_counter = 0 

        self.localizer = Localizer()
        
        # 2. UI Layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame, bg="#222", width=400)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = tk.Frame(self.main_frame, bg="#333", width=800)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas_ref = tk.Canvas(self.left_frame, bg="black")
        self.canvas_ref.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_drone = tk.Canvas(self.right_frame, bg="black", cursor="fleur")
        self.canvas_drone.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(self.left_frame, text="Reference Map", fg="white", bg="#222").pack(side=tk.TOP)
        tk.Label(self.right_frame, text="Drone View (Markers + History)", fg="white", bg="#333").pack(side=tk.TOP)

        self.control_frame = tk.Frame(root, bg="#444", height=80)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.btn_activate = tk.Button(self.control_frame, text="ACTIVATE VNS", 
                                      bg="#00AA00", fg="white", font=("Arial", 12, "bold"),
                                      command=self.activate_vns)
        self.btn_activate.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.X, expand=True)
        
        self.btn_deactivate = tk.Button(self.control_frame, text="DEACTIVATE / CLEAR", 
                                        bg="#AA0000", fg="white", font=("Arial", 12, "bold"),
                                        command=self.deactivate_vns)
        self.btn_deactivate.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.X, expand=True)

        self.canvas_drone.bind("<Button-1>", self.on_mouse_down)
        self.canvas_drone.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas_drone.bind("<MouseWheel>", self.on_mouse_wheel) 
        self.canvas_drone.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas_drone.bind("<Button-5>", self.on_mouse_wheel)

        self.update_drone_view()
        self.update_ref_view()
        self.root.after(100, self.run_localization)

    def activate_vns(self):
        self.vns_active = True
        print("VNS Activated.")

    def deactivate_vns(self):
        self.vns_active = False
        self.gps_markers = [] 
        self.last_marker_pos = None
        self.marker_counter = 0
        self.update_ref_view(predicted_pos=None)
        self.update_drone_view()
        print("VNS Deactivated.")

    def on_mouse_down(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_mouse_drag(self, event):
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.drone_x += dx
        self.drone_y += dy
        self.drone_x = max(50, min(self.map_w - 50, self.drone_x))
        self.drone_y = max(50, min(self.map_h - 50, self.drone_y))
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.update_drone_view()

    def on_mouse_wheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.altitude = max(10, self.altitude - 5)
        else:
            self.altitude = min(200, self.altitude + 5)
        self.update_drone_view()

    def get_dynamic_crop_size(self):
        vw = self.canvas_drone.winfo_width()
        if vw <= 1: vw = 800
        scale_factor = self.altitude / 50.0 
        crop_size = int(SCREEN_BOX_SIZE * scale_factor)
        return crop_size

    def get_input_crop(self):
        box_size = self.get_dynamic_crop_size()
        half_box = box_size // 2
        left = self.drone_x - half_box
        top = self.drone_y - half_box
        right = self.drone_x + half_box
        bottom = self.drone_y + half_box
        
        crop = self.map_high_res.crop((left, top, right, bottom))
        if crop.size[0] != box_size or crop.size[1] != box_size:
             crop = crop.resize((box_size, box_size))
        return crop, box_size

    def get_lat_lon(self, px, py):
        global_xtile = self.tl_xtile + (px / TILE_SIZE)
        global_ytile = self.tl_ytile + (py / TILE_SIZE)
        lat, lon = num2deg(global_xtile, global_ytile, ZOOM_LEVEL)
        return lat, lon

    def update_drone_view(self):
        vw = self.canvas_drone.winfo_width()
        vh = self.canvas_drone.winfo_height()
        if vw <= 1: vw, vh = 800, 600
        
        scale_factor = self.altitude / 50.0 
        crop_w = int(vw * scale_factor)
        crop_h = int(vh * scale_factor)
        
        left = self.drone_x - crop_w // 2
        top = self.drone_y - crop_h // 2
        right = left + crop_w
        bottom = top + crop_h
        
        view_img = self.map_high_res.crop((left, top, right, bottom))
        view_img = view_img.resize((vw, vh), Image.Resampling.NEAREST)

        draw = ImageDraw.Draw(view_img)
        
        # --- DRAW MARKERS ON MAP ---
        for m in self.gps_markers:
            mx, my = m['x'], m['y']
            m_color = m['color']
            
            # Translate to crop local coords
            local_x = (mx - left)
            local_y = (my - top)
            
            # Scale to screen size
            screen_x = local_x * (vw / crop_w)
            screen_y = local_y * (vh / crop_h)
            
            # Draw Marker Circle
            r = 6
            draw.ellipse((screen_x - r, screen_y - r, screen_x + r, screen_y + r), 
                         fill=m_color, outline="white", width=2)
            
            # Draw Coordinates Text Next to Marker
            draw.text((screen_x + 12, screen_y - 10), 
                      f"{m['lat']:.5f}, {m['lon']:.5f}", 
                      fill=m_color, font=self.marker_font, stroke_width=1, stroke_fill="black")

        self.tk_drone_img = ImageTk.PhotoImage(view_img)
        self.canvas_drone.create_image(0, 0, image=self.tk_drone_img, anchor=tk.NW)
        
        cx, cy = vw // 2, vh // 2
        
        # --- DRAW VISUALS ---
        half_vis = SCREEN_BOX_SIZE / 2
        self.canvas_drone.create_rectangle(cx - half_vis, cy - half_vis, 
                                           cx + half_vis, cy + half_vis,
                                           outline="blue", width=2)

        self.canvas_drone.create_line(cx - 15, cy, cx + 15, cy, fill="red", width=2)
        self.canvas_drone.create_line(cx, cy - 15, cx, cy + 15, fill="red", width=2)

        # Altitude Text
        self.canvas_drone.create_text(30, 30, text=f"ALT: {self.altitude}m", 
                                      fill="#00FF00", font=("Arial", 16, "bold"), anchor="w")

        # --- DRAW HISTORY LIST (HUD) ---
        # Display the last 4 recognized points in a list on the left side
        if self.vns_active:
            y_start = 70
            self.canvas_drone.create_text(30, y_start, text="LAST 4 POINTS:", 
                                          fill="white", font=("Arial", 14, "bold"), anchor="w")
            
            # Get last 4 markers
            recent_markers = self.gps_markers[-4:]
            
            for i, m in enumerate(recent_markers):
                txt = f"[{i+1}] {m['lat']:.5f}, {m['lon']:.5f}"
                self.canvas_drone.create_text(30, y_start + 30 + (i*25), text=txt, 
                                              fill=m['color'], font=("Arial", 14, "bold"), anchor="w")

    def update_ref_view(self, predicted_pos=None):
        self.tk_ref_img = ImageTk.PhotoImage(self.map_ref)
        self.canvas_ref.create_image(0, 0, image=self.tk_ref_img, anchor=tk.NW)
        
        if predicted_pos is not None:
            px, py = predicted_pos
            current_map_crop_size = self.get_dynamic_crop_size()
            ref_box_size = (current_map_crop_size * self.ref_scale) / 2
            
            self.canvas_ref.create_rectangle(px - ref_box_size, py - ref_box_size,
                                             px + ref_box_size, py + ref_box_size,
                                             outline="cyan", width=3)
            self.canvas_ref.create_text(px, py - ref_box_size - 10, text="MATCH", fill="cyan", font=("Arial", 10, "bold"))

    def run_localization(self):
        if self.vns_active:
            input_crop, crop_sz = self.get_input_crop()
            input_cv = np.array(input_crop)
            
            if input_cv.shape[0] > 0 and input_cv.shape[1] > 0:
                match_coords = self.localizer.match(self.ref_cv, input_cv)
                
                if match_coords is not None:
                    ref_x, ref_y = match_coords
                    real_x = ref_x / self.ref_scale
                    real_y = ref_y / self.ref_scale
                    
                    lat, lon = self.get_lat_lon(real_x, real_y)
                    
                    # Check distance to last marker
                    add_marker = False
                    if self.last_marker_pos is None:
                        add_marker = True
                    else:
                        lx, ly = self.last_marker_pos
                        dist = math.sqrt((real_x - lx)**2 + (real_y - ly)**2)
                        if dist > MARKER_DISTANCE_THRESHOLD:
                            add_marker = True
                            
                    if add_marker:
                        # Cycle through colors: Red -> Green -> Cyan -> Yellow
                        color = MARKER_COLORS[self.marker_counter % len(MARKER_COLORS)]
                        self.marker_counter += 1
                        
                        self.gps_markers.append({
                            'x': real_x, 'y': real_y, 
                            'lat': lat, 'lon': lon,
                            'color': color
                        })
                        self.last_marker_pos = (real_x, real_y)
                        
                        if len(self.gps_markers) > 50:
                            self.gps_markers.pop(0)

                    self.update_ref_view(predicted_pos=match_coords)
                    self.update_drone_view()
            
        self.root.after(200, self.run_localization)

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneSimApp(root)
    root.mainloop()