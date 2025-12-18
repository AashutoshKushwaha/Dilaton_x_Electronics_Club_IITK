import tkinter as tk
import minimalmodbus
import time
import math

# --- CONFIGURATION ---
PORT = 'COM3'
ADDRESS = 80
BAUD_RATE = 9600

# --- SENSITIVITY TUNING (THE AMPLIFIER) ---
# 1. THRESHOLD: Lowered to 0.03g to catch tiny slides.
# (Because we have a filter now, we can afford to go this low).
NOISE_THRESHOLD = 0.03

# 2. AMPLIFICATION: Multiplies the force to make the cursor faster.
SIGNAL_BOOST = 15.0  

# 3. SMOOTHING (0.0 to 1.0):
# 0.1 = Very Smooth (Less Jitter, slight delay). 
# 0.9 = Raw Data (Fast response, more jitter).
SMOOTHING_FACTOR = 0.2

FRICTION = 0.85
SCALE_FACTOR = 100.0 # Pixel Zoom

# --- SENSOR SETUP ---
def setup_sensor():
    try:
        sensor = minimalmodbus.Instrument(PORT, ADDRESS)
        sensor.serial.baudrate = BAUD_RATE
        sensor.serial.timeout = 0.05
        return sensor
    except:
        return None

sensor = setup_sensor()

def get_signed_int(val):
    return val - 65536 if val > 32767 else val

class SensitiveMouseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("High-Sensitivity IMU Mouse")
        self.root.geometry("1000x850")
        self.root.configure(bg="#1e1e1e")

        self.px, self.py = 500.0, 425.0
        self.vx, self.vy = 0.0, 0.0
        
        # Smoothing Variables (Memory of previous frame)
        self.avg_ax = 0.0
        self.avg_ay = 0.0
        
        # Calibration Offsets
        self.off_ax = 0.0
        self.off_ay = 0.0

        self.setup_ui()
        self.last_time = time.time()
        self.update_loop()

    def setup_ui(self):
        # Stats Bar
        bar = tk.Frame(self.root, bg="#333", pady=10)
        bar.pack(fill=tk.X)
        self.var_stat = tk.StringVar(value="RAW: 0.00 | SMOOTH: 0.00")
        tk.Label(bar, textvariable=self.var_stat, bg="#333", fg="#0f0", font=("Consolas", 12)).pack(side=tk.LEFT, padx=20)
        
        # Buttons
        tk.Button(bar, text="CALIBRATE (Keep Still)", bg="orange", command=self.calibrate).pack(side=tk.RIGHT, padx=20)
        tk.Button(bar, text="RESET", bg="red", fg="white", command=self.reset_pos).pack(side=tk.RIGHT, padx=5)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_grid()
        self.cursor = self.canvas.create_oval(0,0,0,0, fill="cyan", outline="white", width=2)

    def draw_grid(self):
        for i in range(0, 1000, 50): self.canvas.create_line(i, 0, i, 850, fill="#222")
        for i in range(0, 850, 50): self.canvas.create_line(0, i, 1000, i, fill="#222")

    def reset_pos(self):
        self.px, self.py = 500.0, 425.0
        self.vx, self.vy = 0, 0

    def calibrate(self):
        # Take a snapshot of the current SMOOTHED values
        self.off_ax = self.avg_ax
        self.off_ay = self.avg_ay
        self.vx, self.vy = 0, 0
        print(f"Calibrated: Offsets {self.off_ax:.3f}, {self.off_ay:.3f}")

    def read_clean_data(self):
        if not sensor: return 0,0
        try:
            data = sensor.read_registers(52, 9, functioncode=3)
            
            # Raw Accel
            ax = get_signed_int(data[0]) / 32768.0 * 16.0
            ay = get_signed_int(data[1]) / 32768.0 * 16.0
            
            # Angles (for Gravity Removal)
            roll = get_signed_int(data[6]) / 32768.0 * 180.0
            pitch = get_signed_int(data[7]) / 32768.0 * 180.0
            
            # Gravity Math (Subtract the tilt force)
            g_x = math.sin(math.radians(roll))
            g_y = math.sin(math.radians(pitch))
            
            clean_ax = ax - g_x
            clean_ay = ay - g_y
            
            return clean_ax, clean_ay
        except:
            return 0,0

    def update_loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0.1: dt = 0.1

        # 1. GET RAW DATA
        raw_ax, raw_ay = self.read_clean_data()
        
        # 2. LOW PASS FILTER (The Noise Killer)
        # We blend the new data with the old data.
        self.avg_ax = (raw_ax * SMOOTHING_FACTOR) + (self.avg_ax * (1.0 - SMOOTHING_FACTOR))
        self.avg_ay = (raw_ay * SMOOTHING_FACTOR) + (self.avg_ay * (1.0 - SMOOTHING_FACTOR))

        # 3. APPLY CALIBRATION
        # We use the Smoothed value minus the Calibration offset
        eff_ax = self.avg_ax - self.off_ax
        eff_ay = self.avg_ay - self.off_ay

        # 4. THRESHOLD CHECK
        mag = math.sqrt(eff_ax**2 + eff_ay**2)
        
        if mag < NOISE_THRESHOLD:
            # STOPPED
            eff_ax = 0
            eff_ay = 0
            self.vx = 0 # Aggressive stop
            self.vy = 0
            self.canvas.itemconfig(self.cursor, outline="gray")
        else:
            # MOVING - APPLY BOOST
            # This is the "Software Amplifier"
            eff_ax *= SIGNAL_BOOST
            eff_ay *= SIGNAL_BOOST
            self.canvas.itemconfig(self.cursor, outline="white")

        # 5. PHYSICS INTEGRATION
        self.vx += eff_ax * 9.81 * dt
        self.vy += eff_ay * 9.81 * dt
        
        self.vx *= FRICTION
        self.vy *= FRICTION
        
        self.px += self.vx * SCALE_FACTOR * dt
        self.py -= self.vy * SCALE_FACTOR * dt # Inverted Y

        # 6. UPDATE VISUALS
        self.var_stat.set(f"RAW: {raw_ax:.3f} | FILTERED: {eff_ax:.3f}")
        
        r = 10
        self.canvas.coords(self.cursor, self.px-r, self.py-r, self.px+r, self.py+r)
        
        self.root.after(20, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = SensitiveMouseApp(root)
    root.mainloop()