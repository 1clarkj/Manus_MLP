import socket
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
import threading
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


# UDP server setup
UDP_IP = "192.168.0.115"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Listening on {UDP_IP}:{UDP_PORT}")  

while True:
    try:
        data, addr = sock.recvfrom(1024)
        input_data = np.frombuffer(data, dtype=np.float32).reshape(1, -1)

        # Apply the same normalization
        input_data_scaled = scaler.transform(input_data)

        # Predict and print softmax probabilities
        probs = model.predict_proba(input_data_scaled)
        print(f"Softmax Probabilities: {probs}")
    except Exception as e:
        print(f"Error: {e}")
