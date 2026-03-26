import argparse
import json
import socket
from pathlib import Path

import joblib
import numpy as np
import warnings


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

GESTURE_TO_COMMAND = {
    "pointing": 1,
    "thumbs_up": 2,
}


def load_class_mapping():
    mapping_path = ARTIFACTS / "class_mapping.json"
    if not mapping_path.exists():
        return {0: "pointing", 1: "thumbs_up", 2: "okay_sign"}

    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def resolve_model_path():
    primary = ARTIFACTS / "mlp_model_final.pkl"
    fallback = ARTIFACTS / "mlp_model.pkl"
    if primary.exists():
        return primary
    return fallback


def decode_prediction(probabilities, class_mapping, threshold):
    class_id = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    gesture_name = class_mapping.get(class_id, f"class_{class_id}")
    command = GESTURE_TO_COMMAND.get(gesture_name, 0) if confidence >= threshold else 0
    normalized = gesture_name.strip().lower()
    reset_flag = confidence >= threshold and (
        ("reset" in normalized) or (normalized in {"okay_sign", "ok_sign", "okay"})
    )
    return class_id, gesture_name, confidence, command, reset_flag


def main():
    parser = argparse.ArgumentParser(
        description="Real-time UDP gesture inference (no GUI, no Pi control loop)."
    )
    parser.add_argument("--ip", default="127.0.0.1", help="UDP listen IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5005, help="UDP listen port (default: 5005)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.93,
        help="Confidence threshold for command mapping (default: 0.93)",
    )
    args = parser.parse_args()

    model_path = resolve_model_path()
    scaler_path = ARTIFACTS / "minmax_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    class_mapping = load_class_mapping()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))

    print(f"Listening for hand data on {args.ip}:{args.port}")
    print(f"Using model: {model_path.name}")
    print("Expected packet format: 40 float32 values (20 left + 20 right).")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            data, _ = sock.recvfrom(2048)
            floats = np.frombuffer(data, dtype=np.float32)

            if floats.shape[0] != 40:
                print(f"Skipping packet: expected 40 floats, got {floats.shape[0]}")
                continue

            left_hand = floats[:20].reshape(1, -1)
            right_hand = floats[20:].reshape(1, -1)

            left_scaled = scaler.transform(left_hand)
            right_scaled = scaler.transform(right_hand)

            left_probs = model.predict_proba(left_scaled)[0]
            right_probs = model.predict_proba(right_scaled)[0]

            l_id, l_name, l_conf, x_cmd, l_reset = decode_prediction(
                left_probs, class_mapping, args.threshold
            )
            r_id, r_name, r_conf, y_cmd, r_reset = decode_prediction(
                right_probs, class_mapping, args.threshold
            )
            reset_flag = l_reset or r_reset

            print(
                f"Left: {l_name} ({l_conf:.3f}, class={l_id}) | "
                f"Right: {r_name} ({r_conf:.3f}, class={r_id}) | "
                f"Mapped command (x,y)=({x_cmd},{y_cmd}) | "
                f"reset_flag={int(reset_flag)}"
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
