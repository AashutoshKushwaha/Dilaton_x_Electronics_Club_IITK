# IMU High-Sensitivity Motion Tracker & Virtual Mouse

## 📖 Project Overview
This project transforms a standard industrial IMU sensor (specifically WitMotion Modbus series) into a high-sensitivity, physics-based input device. 

Unlike basic tilt sensors that map angle directly to position, this system calculates **linear displacement**. It uses accelerometer data, compensates for gravity tilt, applies signal processing (Low Pass Filters), and uses a custom physics engine to simulate momentum and friction. The result is a cursor that behaves like a physical puck sliding on a surface.

## 🚀 Features
* **Gravity Compensation:** Mathematically removes the gravity vector so the cursor moves based on lateral slides, not just tilting.
* **Noise Filtration:** Implements a software-based Low Pass Filter (LPF) to smooth out sensor jitter.
* **Physics Engine:** Simulates velocity, acceleration, and friction for organic, weighted movement.
* **Drift Calibration:** "Zero-motion" calibration feature to eliminate drift caused by sensor bias.
* **Visual Dashboard:** A Python Tkinter GUI that visualizes the cursor, raw data, and grid position.

---

## 🛠️ System Requirements

### Hardware
* **IMU Sensor:** WitMotion HWT905 (or similar Modbus-over-RS485 sensor).
* **Interface:** USB-to-RS485 converter (e.g., CH340 or FTDI chip).
* **PC:** Windows (tested), Linux, or macOS.

### Software
* Python 3.x
* Libraries: `minimalmodbus`, `pyserial`, `tkinter` (standard).

---

## 📦 Installation & Setup

1.  **Install Python Dependencies:**
    Open your terminal or command prompt and run:
    ```bash
    pip install minimalmodbus pyserial
    ```

2.  **Identify Connection Port:**
    * **Windows:** Open Device Manager > Ports (COM & LPT). Look for "USB-SERIAL CH340" (or similar). Note the COM port (e.g., `COM3`).
    * **Linux/Mac:** Look in `/dev/` (e.g., `/dev/ttyUSB0`).

3.  **Update Configuration:**
    Open `imumotion.py` and edit the configuration section at the top:
    ```python
    PORT = 'COM3'       # Change to your actual port
    ADDRESS = 80        # Default WitMotion Modbus Address (0x50)
    BAUD_RATE = 9600    # Default Baud Rate
    ```

---

## ⚙️ Configuration & Tuning Guide

The feel of the mouse is controlled by specific variables in the script. Adjust these to suit your sensor's sensitivity.

| Variable | Default | Function |
| :--- | :--- | :--- |
| **`NOISE_THRESHOLD`** | `0.03` | **The Deadzone.** The sensor ignores acceleration below this value (in g). <br>• *Increase* if cursor drifts while standing still.<br>• *Decrease* for higher sensitivity to tiny movements. |
| **`SIGNAL_BOOST`** | `15.0` | **The Amplifier.** Multiplies the physical force applied to the cursor.<br>• *Higher* = Cursor moves faster with less effort. |
| **`SMOOTHING_FACTOR`** | `0.2` | **The Filter.** Controls the blend between new data and old data.<br>• `0.1`: Very smooth, but "laggy".<br>• `0.9`: Very responsive, but jittery (noisy). |
| **`FRICTION`** | `0.85` | **The Brakes.** How fast the cursor loses speed.<br>• `0.99`: Like sliding on ice.<br>• `0.60`: Like sliding on mud. |

---

## 🧮 How It Works (The Math)

### 1. Gravity Compensation
Accelerometers always read 1g (gravity) pointing down. If you tilt the sensor, the X/Y axis "sees" some of that gravity, which looks like movement. We remove this using the Angle (Roll/Pitch) data:

$$ A_{clean} = A_{raw} - \sin(\theta_{tilt}) $$

This ensures the cursor only moves when you actually *slide* the sensor.

### 2. The Low Pass Filter
Sensors are noisy. We use a simple infinite impulse response (IIR) filter to smooth the data:

$$ \text{Avg}_{new} = (\text{Raw} \times \text{Factor}) + (\text{Avg}_{old} \times (1 - \text{Factor})) $$

### 3. Physics Integration
We use semi-implicit Euler integration to update the cursor position:
1.  **Velocity Update:** $V = V + (A \times 9.81 \times dt)$
2.  **Friction Apply:** $V = V \times \text{Friction}$
3.  **Position Update:** $P = P + V \times \text{Scale}$

---

## 🎮 Controls & Usage

1.  **Start the Application:**
    Run the script: `python imumotion.py`
    
2.  **Calibrate (CRITICAL STEP):**
    * Place the sensor flat on your desk.
    * **Do not touch it.**
    * Click the Orange **CALIBRATE** button.
    * *Why?* All sensors have a slight manufacturing bias. This step measures that bias and subtracts it permanently.

3.  **Reset:**
    If the cursor moves too fast and flies off the canvas, click the Red **RESET** button to bring it back to the center (500, 425).

---

## 📂 Code Structure

* **`setup_sensor()`**: Establishes the serial connection via Modbus.
* **`SensitiveMouseApp` (Class)**: The main GUI application.
    * **`update_loop()`**: The "game loop" running every ~20ms. It handles reading data, physics calculations, and redrawing the canvas.
    * **`read_clean_data()`**: Fetches raw registers and performs the math to remove gravity.
    * **`calibrate()`**: Captures the current sensor reading and stores it as the "zero" offset.

---

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Crash on startup** | Check your `PORT`. Ensure the USB is plugged in. Ensure `minimalmodbus` is installed. |
| **Cursor drifts constantly** | You moved the sensor during startup. Place it flat and click **CALIBRATE**. |
| **Cursor is jittery** | Decrease `SMOOTHING_FACTOR` (try 0.1) or increase `NOISE_THRESHOLD`. |
| **Cursor is too slow** | Increase `SIGNAL_BOOST` (try 25.0). |
