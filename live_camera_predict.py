import os
import time
import cv2
import numpy as np

# 1. Load model (prefer TFLite for fast real-time inference, fallback to .pkl)
TFLITE_MODEL_PATH = "mosquito_model.tflite"
PKL_MODEL_PATH = "mosquito_model.pkl"
CLASS_NAMES = ["Aedes aegypti", "Aedes albopictus"]

print("Initializing Mosquito Classifier for Live Camera...")

use_tflite = False
interpreter = None
keras_model = None

if os.path.exists(TFLITE_MODEL_PATH):
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        use_tflite = True
        print(f"Loaded TFLite model from '{TFLITE_MODEL_PATH}'")
    except Exception as e:
        print(f"Could not initialize TFLite: {e}. Falling back to Keras model.")

if not use_tflite and os.path.exists(PKL_MODEL_PATH):
    import pickle
    from tensorflow.keras.models import model_from_json
    with open(PKL_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    keras_model = model_from_json(model_data['architecture'])
    keras_model.set_weights(model_data['weights'])
    print(f"Loaded Keras model from '{PKL_MODEL_PATH}'")

if not use_tflite and keras_model is None:
    raise FileNotFoundError("Neither 'mosquito_model.tflite' nor 'mosquito_model.pkl' could be loaded.")

def predict_crop(crop_bgr):
    """Preprocess a 64x64 BGR crop and return (class_name, confidence_pct, probs)"""
    img_resized = cv2.resize(crop_bgr, (64, 64))
    img_normalized = (img_resized.astype(np.float32) / 255.0).reshape(1, 64, 64, 3)

    if use_tflite:
        interpreter.set_tensor(input_details[0]['index'], img_normalized)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        output = keras_model.predict(img_normalized, verbose=0)[0]

    class_idx = int(np.argmax(output))
    confidence = float(output[class_idx]) * 100.0
    return CLASS_NAMES[class_idx], confidence, output

def main():
    print("Opening webcam (Index 0)... Press 'q' to exit.")
    cap = cv2.VideoCapture(0)

    # If camera 0 cannot be opened, try camera 1
    if not cap.isOpened():
        print("Camera 0 failed, trying Camera 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("\n[ERROR] No camera detected! Please check that your webcam is connected and not in use by another app.")
        return

    # Set camera resolution (standard 640x480 or 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    use_roi = True
    prev_time = time.time()
    fps = 0.0

    window_name = "Mosquito Species Classifier - Live Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        h, w, _ = frame.shape

        # Define Region of Interest (ROI) box in the center (square)
        box_size = int(min(h, w) * 0.6)
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        if use_roi:
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame

        # Run inference
        class_name, confidence, probs = predict_crop(crop)

        # FPS calculation
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-5))
        prev_time = now

        # Draw UI
        display = frame.copy()

        # Color coding: Green/Cyan for Aegypti, Orange/Blue for Albopictus
        box_color = (0, 220, 100) if "aegypti" in class_name.lower() else (255, 140, 0)

        if use_roi:
            # Draw target box corners
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(display, "TARGET AREA (Place mosquito here)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Top overlay bar for readability
        cv2.rectangle(display, (0, 0), (w, 85), (20, 20, 20), -1)

        # Title / Prediction text
        label_text = f"{class_name.upper()}: {confidence:.1f}%"
        cv2.putText(display, label_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, box_color, 2, cv2.LINE_AA)

        # Probabilities breakdown
        prob_text = f"Aegypti: {probs[0]*100:.1f}%  |  Albopictus: {probs[1]*100:.1f}%  |  FPS: {fps:.1f}"
        cv2.putText(display, prob_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom help bar
        cv2.rectangle(display, (0, h - 30), (w, h), (20, 20, 20), -1)
        help_text = "[Q] Quit  |  [S] Save Snapshot  |  [T] Toggle Target Box"
        cv2.putText(display, help_text, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('t'):  # Toggle ROI box vs full frame
            use_roi = not use_roi
            print(f"Target box mode: {'Enabled (center)' if use_roi else 'Disabled (full frame)'}")
        elif key == ord('s'):  # Save screenshot
            screenshot_count += 1
            filename = f"live_capture_{screenshot_count}_{class_name.replace(' ', '_')}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved snapshot to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")

if __name__ == "__main__":
    main()
