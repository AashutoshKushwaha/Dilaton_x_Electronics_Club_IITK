import tkinter as tk
from tkinter import ttk, messagebox
import minimalmodbus
import serial
import csv
import os
from datetime import datetime

# --- CONFIGURATION ---
PORT = 'COM3'         # Change this to your actual COM port
ADDRESS = 0x50        # WitMotion Default Address
BAUD_RATE = 9600

# --- GLOBAL VARIABLES FOR LOGGING ---
is_recording = False
csv_file = None
csv_writer = None
start_time = None

# --- CONNECT TO SENSOR ---
def setup_sensor():
    try:
        sensor = minimalmodbus.Instrument(PORT, ADDRESS)
        sensor.serial.baudrate = BAUD_RATE
        sensor.serial.timeout = 0.1
        sensor.clear_buffers_before_each_transaction = True
        return sensor
    except Exception as e:
        return None

sensor = setup_sensor()

# --- CSV LOGGING FUNCTIONS ---
def toggle_recording():
    global is_recording, csv_file, csv_writer, start_time
    
    if not is_recording:
        # START RECORDING
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"SensorData_{timestamp}.csv"
            
            # Open file in write mode
            csv_file = open(filename, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            
            # Write Header
            header = ["Timestamp", "Runtime_Sec", 
                      "Acc_X(g)", "Acc_Y(g)", "Acc_Z(g)", 
                      "Gyro_X(deg/s)", "Gyro_Y(deg/s)", "Gyro_Z(deg/s)", 
                      "Roll(deg)", "Pitch(deg)", "Yaw(deg)", 
                      "Temp(C)"]
            csv_writer.writerow(header)
            
            # Update State
            is_recording = True
            start_time = datetime.now()
            btn_record.config(text="STOP RECORDING", bg="#dc3545", fg="white") # Red button
            lbl_rec_status.config(text=f"● REC: {filename}", foreground="red")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not create file: {e}")
    else:
        # STOP RECORDING
        is_recording = False
        if csv_file:
            csv_file.close()
            csv_file = None
        
        btn_record.config(text="START RECORDING", bg="#28a745", fg="white") # Green button
        lbl_rec_status.config(text="● Ready", foreground="gray")

def log_data_to_csv(data_tuple):
    if is_recording and csv_writer:
        try:
            now = datetime.now()
            runtime = (now - start_time).total_seconds()
            
            # Create a list for the CSV row: [Time, Runtime, Data...]
            row = [now.strftime("%H:%M:%S.%f")[:-3], f"{runtime:.3f}"] + list(data_tuple)
            
            csv_writer.writerow(row)
            # Optional: Flush periodically to ensure data is saved if crash occurs
            # csv_file.flush() 
        except Exception as e:
            print(f"Write Error: {e}")

# --- DATA PROCESSING ---
def get_signed_int(val):
    if val > 32767: return val - 65536
    return val

def read_all_data():
    if sensor is None: return None
    try:
        # Read 13 registers (Acc, Gyro, Angle, Temp)
        data = sensor.read_registers(52, 13, functioncode=3)
        
        # Calculations
        ax = get_signed_int(data[0]) / 32768.0 * 16.0
        ay = get_signed_int(data[1]) / 32768.0 * 16.0
        az = get_signed_int(data[2]) / 32768.0 * 16.0
        gx = get_signed_int(data[3]) / 32768.0 * 2000.0
        gy = get_signed_int(data[4]) / 32768.0 * 2000.0
        gz = get_signed_int(data[5]) / 32768.0 * 2000.0
        roll = get_signed_int(data[6]) / 32768.0 * 180.0
        pitch = get_signed_int(data[7]) / 32768.0 * 180.0
        yaw = get_signed_int(data[8]) / 32768.0 * 180.0
        temp = get_signed_int(data[12]) / 100.0
        
        return (ax, ay, az, gx, gy, gz, roll, pitch, yaw, temp)
    except:
        return None

# --- GUI UPDATE LOOP ---
def update_dashboard():
    data = read_all_data()
    
    if data:
        # 1. Update CSV
        log_data_to_csv(data)
        
        # 2. Update GUI
        ax, ay, az, gx, gy, gz, r, p, y, t = data
        v_acc_x.set(f"{ax:.3f}")
        v_acc_y.set(f"{ay:.3f}")
        v_acc_z.set(f"{az:.3f}")
        v_gyro_x.set(f"{gx:.1f}")
        v_gyro_y.set(f"{gy:.1f}")
        v_gyro_z.set(f"{gz:.1f}")
        v_ang_x.set(f"{r:.2f}")
        v_ang_y.set(f"{p:.2f}")
        v_ang_z.set(f"{y:.2f}")
        v_temp.set(f"{t:.1f} °C")
        
        lbl_conn_status.config(text="Link: Active", foreground="green")
    else:
        lbl_conn_status.config(text="Link: Error", foreground="red")

    root.after(10, update_dashboard)

# --- CLEANUP ON CLOSE ---
def on_closing():
    if is_recording and csv_file:
        csv_file.close()
    root.destroy()

# --- GUI LAYOUT ---
root = tk.Tk()
root.title("HWT905 Logger")
root.geometry("750x550")
root.protocol("WM_DELETE_WINDOW", on_closing) # Handle X button click

# Styles
style = ttk.Style()
style.theme_use('clam')
style.configure("Card.TFrame", background="white", relief="groove", borderwidth=2)
style.configure("Data.TLabel", background="white", font=("Consolas", 14, "bold"))

# Header
frame_top = tk.Frame(root, bg="#333", height=50)
frame_top.pack(fill=tk.X)
tk.Label(frame_top, text="SENSOR LOGGER", fg="white", bg="#333", font=("Arial", 14, "bold")).pack(pady=10)

# Control Bar (Record Button)
frame_ctrl = tk.Frame(root, bg="#eee", pady=10)
frame_ctrl.pack(fill=tk.X)

btn_record = tk.Button(frame_ctrl, text="START RECORDING", bg="#28a745", fg="white", 
                       font=("Arial", 12, "bold"), width=20, command=toggle_recording)
btn_record.pack(side=tk.LEFT, padx=20)

lbl_rec_status = tk.Label(frame_ctrl, text="● Ready", bg="#eee", fg="gray", font=("Arial", 10))
lbl_rec_status.pack(side=tk.LEFT, padx=10)

lbl_conn_status = tk.Label(frame_ctrl, text="Link: Connecting...", bg="#eee", fg="gray")
lbl_conn_status.pack(side=tk.RIGHT, padx=20)

# Data Grid
container = ttk.Frame(root)
container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Helper for layout
def make_card(parent, title, r, c):
    f = ttk.Frame(parent, style="Card.TFrame", padding=10)
    f.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
    ttk.Label(f, text=title, background="white", font=("Arial", 10, "bold"), foreground="#777").pack(anchor="w")
    return f

def make_row(parent, label, var):
    row = ttk.Frame(parent, style="Card.TFrame") # Inner frame just for packing
    row.configure(relief="flat", borderwidth=0)
    row.pack(fill=tk.X, pady=2)
    ttk.Label(row, text=label, background="white", width=6, foreground="#555").pack(side=tk.LEFT)
    ttk.Label(row, textvariable=var, style="Data.TLabel").pack(side=tk.RIGHT)

# Variables
v_acc_x, v_acc_y, v_acc_z = tk.StringVar(), tk.StringVar(), tk.StringVar()
v_gyro_x, v_gyro_y, v_gyro_z = tk.StringVar(), tk.StringVar(), tk.StringVar()
v_ang_x, v_ang_y, v_ang_z = tk.StringVar(), tk.StringVar(), tk.StringVar()
v_temp = tk.StringVar()

container.columnconfigure((0,1,2), weight=1)

# Card 1: Angle
c1 = make_card(container, "ANGLE (deg)", 0, 0)
make_row(c1, "Roll", v_ang_x)
make_row(c1, "Pitch", v_ang_y)
make_row(c1, "Yaw", v_ang_z)

# Card 2: Gyro
c2 = make_card(container, "GYRO (deg/s)", 0, 1)
make_row(c2, "X", v_gyro_x)
make_row(c2, "Y", v_gyro_y)
make_row(c2, "Z", v_gyro_z)

# Card 3: Accel
c3 = make_card(container, "ACCEL (g)", 0, 2)
make_row(c3, "X", v_acc_x)
make_row(c3, "Y", v_acc_y)
make_row(c3, "Z", v_acc_z)

# Temp Display
f_temp = tk.Frame(root, bg="#eee", height=30)
f_temp.pack(fill=tk.X, side=tk.BOTTOM)
tk.Label(f_temp, text="Temperature:", bg="#eee").pack(side=tk.RIGHT)
tk.Label(f_temp, textvariable=v_temp, bg="#eee", font=("Arial", 10, "bold")).pack(side=tk.RIGHT, padx=10)

root.after(500, update_dashboard)
root.mainloop()