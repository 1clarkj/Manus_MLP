import socket
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
import threading
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

# Load trained model and scaler
model = joblib.load("mlp_model1.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# Maze dimensions in mm
MAX_WIDTH_MM = 355
MAX_HEIGHT_MM = 285
TARGET_ASPECT = MAX_WIDTH_MM / MAX_HEIGHT_MM  

# Initial ball position
ball_x, ball_y = 177.5, 142.5
ball_pos_mm = [ball_x, ball_y]  # center
ball_radius_mm = 20
step_size_mm = 1.0

# Normalize
ball_pos = [ball_pos_mm[0] / MAX_WIDTH_MM, ball_pos_mm[1] / MAX_HEIGHT_MM]
ball_radius = ball_radius_mm / MAX_WIDTH_MM  # scaled on x-axis
step_size = step_size_mm / MAX_WIDTH_MM

# Each wall is defined as (x1, y1, x2, y2)
maze_walls_mm = [
    (0,0,0,285),  # left
    (0,285,355,285),  # bottom
    (355,285,355,0),  # right
    (355,0,0,0),  # top
    (195,285,195,205),
    (190,210,280,210),
    (355,142.5,115,142.5),  
    (114.5,227.5,114.5,57.5),
    (119.5,62.5,84.5,62.5),
    (119.5,232.5,84.5,232.5),
    (220,142,220,112),
    (200,0,200,60),
    (195,55,285,55),
    (280,50, 280, 95),	
]

# Convert to % of physical space
maze_walls = [
    (
        x1 / MAX_WIDTH_MM,
        y1 / MAX_HEIGHT_MM,
        x2 / MAX_WIDTH_MM,
        y2 / MAX_HEIGHT_MM
    )
    for (x1, y1, x2, y2) in maze_walls_mm
]

# GUI update lock
position_lock = threading.Lock()

# # UDP server setup
# UDP_IP = "127.0.0.1"
# UDP_PORT = 5005
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((UDP_IP, UDP_PORT))
# print(f"Listening on {UDP_IP}:{UDP_PORT}")  

# while True:
#     try:
#         data, addr = sock.recvfrom(1024)
#         input_data = np.frombuffer(data, dtype=np.float32).reshape(1, -1)

#         # Apply the same normalization
#         input_data_scaled = scaler.transform(input_data)

#         # Predict and print softmax probabilities
#         probs = model.predict_proba(input_data_scaled)
#         print(f"Softmax Probabilities: {probs}")
#     except Exception as e:
#         print(f"Error: {e}")


def resize_canvas(event):
    w, h = event.width, event.height
    current_aspect = w / h

    if current_aspect > TARGET_ASPECT:
        new_width = int(h * TARGET_ASPECT)
        canvas.config(width=new_width, height=h)
    else:
        new_height = int(w / TARGET_ASPECT)
        canvas.config(width=w, height=new_height)

    draw_maze()
    draw_ball()

def update_gui():
    draw_maze()
    draw_ball()
    root.after(50, update_gui)

def apply_ip():
    global send_ip, send_port
    send_ip = ip_entry.get()
    send_port = int(port_entry.get())

def draw_maze():
    canvas.delete("maze")
    w, h = canvas.winfo_width(), canvas.winfo_height()
    thickness_px = get_wall_thickness_px()

    for wall in maze_walls:
        x1, y1, x2, y2 = wall
        canvas.create_line(
            x1 * w, y1 * h, x2 * w, y2 * h,
            fill="black", width=thickness_px, tags="maze"
        )

def draw_ball():
    canvas.delete("ball")
    w, h = canvas.winfo_width(), canvas.winfo_height()
    x, y = ball_pos
    r = ball_radius
    canvas.create_oval(
        (x - r) * w, (y - r) * h, (x + r) * w, (y + r) * h,
        fill="blue", tags="ball"
    )

def get_wall_thickness_px():
    wall_thickness_mm = 10
    w_px = canvas.winfo_width()
    h_px = canvas.winfo_height()
    px_x = wall_thickness_mm * (w_px / MAX_WIDTH_MM)
    px_y = wall_thickness_mm * (h_px / MAX_HEIGHT_MM)
    return int((px_x + px_y) / 2)  # average thickness in pixels

# Set up GUI
root = tk.Tk()
root.title("Maze Control GUI")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0)

canvas = tk.Canvas(main_frame, width=300, height=350, bg="white")
canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")
canvas.bind("<Configure>", resize_canvas)

# Allow row and column expansion
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

# IP and Port input
ttk.Label(main_frame, text="Send IP:").grid(row=1, column=0, sticky="e")
ip_entry = ttk.Entry(main_frame)
ip_entry.insert(0, "127.0.0.1")
ip_entry.grid(row=1, column=1)

ttk.Label(main_frame, text="Send Port:").grid(row=2, column=0, sticky="e")
port_entry = ttk.Entry(main_frame)
port_entry.insert(0, "6006")
port_entry.grid(row=2, column=1)

ttk.Button(main_frame, text="Apply", command=apply_ip).grid(row=3, column=0, columnspan=2)

# Set up UDP sender
sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Start GUI loop
update_gui()
root.mainloop()

