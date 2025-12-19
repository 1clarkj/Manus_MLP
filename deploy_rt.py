import socket
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
import threading
import warnings
import queue
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

# Load trained model and scaler
model = joblib.load("mlp_model1.pkl")
scaler = joblib.load("minmax_scaler.pkl")

#Raspberry Pi IP and port for sending hand data via UDP

raspi_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# Maze dimensions in mm
MAX_WIDTH_MM = 355
MAX_HEIGHT_MM = 285
TARGET_ASPECT = MAX_WIDTH_MM / MAX_HEIGHT_MM  

# Initial ball position
ball_x, ball_y = 300, 42
actual_pos=ball_pos_mm = [ball_x, ball_y]  # center
ball_radius_mm = 20
step_size_mm = 7.0

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

class CollisionFeedbackSender:
    def __init__(self, target_ip="127.0.0.1", target_port=6006):
        self.target_ip = target_ip
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.msg_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self.thread.start()

    def send_flag(self, flag_bytes: bytes):
        self.msg_queue.put(flag_bytes)

    def _feedback_loop(self):
        while self.running:
            try:
                msg = self.msg_queue.get(timeout=1)
                self.sock.sendto(msg, (self.target_ip, self.target_port))
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.thread.join()


def listen_for_actual_position():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 6007))  # Listen for position from the Pi

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            pos = np.frombuffer(data, dtype=np.float32)
            if pos.shape == (2,):
                print(f"Received actual position: {pos}")
                 # Convert Pi coordinates (center-origin, mm) → GUI normalized (top-left origin)
                x_mm = pos[0] + (MAX_WIDTH_MM / 2)
                y_mm = (MAX_HEIGHT_MM / 2) - pos[1]

                with position_lock:
                    actual_pos[0] = x_mm / MAX_WIDTH_MM
                    actual_pos[1] = y_mm / MAX_HEIGHT_MM
        except Exception as e:
            print(f"Error in actual position receive: {e}")

def udp_listener():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for hand data on {UDP_IP}:{UDP_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            floats = np.frombuffer(data, dtype=np.float32)
            raspi_socket.sendto(floats.tobytes(), (send_ip, send_port))


            if floats.shape[0] != 40:
                print(f"Invalid input: got {floats.shape[0]} floats")
                continue

            left_hand = floats[:20].reshape(1, -1)
            right_hand = floats[20:].reshape(1, -1)

            # Normalize
            left_scaled = scaler.transform(left_hand)
            right_scaled = scaler.transform(right_hand)

            # Predict and threshold
            left_probs = model.predict_proba(left_scaled)[0]
            right_probs = model.predict_proba(right_scaled)[0]

            # Apply threshold
            x_class = np.argmax(left_probs) + 1 if np.max(left_probs) >= 0.93 else 0
            y_class = np.argmax(right_probs) + 1 if np.max(right_probs) >= 0.93 else 0

            try_move_ball(x_class, y_class)

        except Exception as e:
            print(f"UDP listener error: {e}")


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
    draw_balls()

def update_gui():
    draw_maze()
    draw_balls()
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

def reset_ball_position():
    with position_lock:
        ball_pos[0] = ball_x / MAX_WIDTH_MM
        ball_pos[1] = ball_y / MAX_HEIGHT_MM
    print("Ball position reset to initial coordinates.")

def draw_balls():
    canvas.delete("ball")
    w, h = canvas.winfo_width(), canvas.winfo_height()
    r = ball_radius

    with position_lock:
        x1, y1 = ball_pos
        x2, y2 = actual_pos

    # Predicted (blue)
    canvas.create_oval(
        (x1 - r) * w, (y1 - r) * h, (x1 + r) * w, (y1 + r) * h,
        fill="blue", tags="ball"
    )
    # Actual (green)
    canvas.create_oval(
        (x2 - r) * w, (y2 - r) * h, (x2 + r) * w, (y2 + r) * h,
        fill="green", tags="ball"
    )

def get_wall_thickness_px():
    wall_thickness_mm = 10
    w_px = canvas.winfo_width()
    h_px = canvas.winfo_height()
    px_x = wall_thickness_mm * (w_px / MAX_WIDTH_MM)
    px_y = wall_thickness_mm * (h_px / MAX_HEIGHT_MM)
    return int((px_x + px_y) / 2)  # average thickness in pixels

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return ((px - x1)**2 + (py - y1)**2)**0.5

    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / (dx**2 + dy**2)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((px - proj_x)**2 + (py - proj_y)**2)**0.5

def hits_wall(x, y):
    for x1, y1, x2, y2 in maze_walls:
        if point_to_segment_dist(x, y, x1, y1, x2, y2) <= ball_radius:
            return True
    return False

def try_move_ball(x_dir, y_dir):
    dx = (step_size if x_dir == 1 else -step_size if x_dir == 2 else 0)
    dy = (step_size if y_dir == 2 else -step_size if y_dir == 1 else 0)

    new_x = ball_pos[0] + dx
    new_y = ball_pos[1] + dy

    # Clamp to within bounds
    new_x = max(ball_radius, min(1 - ball_radius, new_x))
    new_y = max(ball_radius, min(1 - ball_radius, new_y))

    x_collides = hits_wall(new_x, ball_pos[1])
    y_collides = hits_wall(ball_pos[0], new_y)

    # If either direction collides, don't update that axis
    with position_lock:
        if not x_collides:
            ball_pos[0] = new_x
        if not y_collides:
            ball_pos[1] = new_y

    # Add: collision from actual position
    with position_lock:
        actual_x, actual_y = actual_pos
    actual_collides = hits_wall(actual_x, actual_y)

    # Send collision signal to C++
    if x_collides or actual_collides:
        feedback_sender.send_flag(b"1")  # Left hand vibration
        print("Collision detected on X axis")
    elif y_collides or actual_collides:
        feedback_sender.send_flag(b"2")  # Right hand vibration
        print("Collision detected on Y axis")
    else:
        feedback_sender.send_flag(b"0")  # No collision
        


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
ip_entry.insert(0, "192.168.0.139")
ip_entry.grid(row=1, column=1)

ttk.Label(main_frame, text="Send Port:").grid(row=2, column=0, sticky="e")
port_entry = ttk.Entry(main_frame)
port_entry.insert(0, "5006")
port_entry.grid(row=2, column=1)

ttk.Button(main_frame, text="Apply", command=apply_ip).grid(row=3, column=0, columnspan=2)
ttk.Button(main_frame, text="Reset Ball", command=reset_ball_position).grid(row=4, column=0, columnspan=2, pady=(10, 0))

# Set up UDP sender
sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

feedback_sender = CollisionFeedbackSender("127.0.0.1", 6006)
udp_thread = threading.Thread(target=udp_listener, daemon=True)
udp_thread.start()

position_thread = threading.Thread(target=listen_for_actual_position, daemon=True)
position_thread.start()

# Start GUI loop
update_gui()
root.mainloop()

