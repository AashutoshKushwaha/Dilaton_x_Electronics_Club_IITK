import os
import cv2
import yaml

# -------------------------
# Load configuration
# -------------------------
CONFIG_PATH = "configs/config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

satellite_dir = cfg["satellite_map_dir"]
video_path = cfg["drone_video_path"]

print("[INFO] Configuration loaded")

# -------------------------
# Check satellite images
# -------------------------
if not os.path.exists(satellite_dir):
    raise FileNotFoundError(f"Satellite directory not found: {satellite_dir}")

sat_files = os.listdir(satellite_dir)
if len(sat_files) == 0:
    raise RuntimeError("No satellite images found")

print(f"[OK] Found {len(sat_files)} satellite image(s): {sat_files}")

sat_img_path = os.path.join(satellite_dir, sat_files[0])
sat_img = cv2.imread(sat_img_path)

if sat_img is None:
    raise RuntimeError("Failed to load satellite image")

print(f"[OK] Satellite image shape: {sat_img.shape}")

# -------------------------
# Check drone video
# -------------------------
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Drone video not found: {video_path}")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    raise RuntimeError("Failed to read first frame from video")

print(f"[OK] Drone video frame shape: {frame.shape}")

cap.release()

print("\n✅ STEP 1 COMPLETE: Data paths and files are valid")
