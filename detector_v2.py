import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# === SETTINGS ===
MODEL_PATH = r"C:\Aarav Things\MLProject\BottleClassifier.keras"
PORT = "COM5"
BAUD_RATE = 9600
CONFIDENCE_THRESHOLD = 0.30  # Very strict
FRAMES_REQUIRED = 10  # More frames needed
SEND_COOLDOWN = 10

# === LOAD MODEL ===
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# === ARDUINO CONNECTION ===
try:
    arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[INFO] Connected to Arduino on {PORT}")
except Exception as e:
    print(f"[WARNING] Could not connect to Arduino: {e}")
    arduino = None

# === CAMERA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Moving average for stability
confidence_history = []
HISTORY_SIZE = 5

recycle_frames = 0
last_send_time = 0

print("[INFO] Detection active. Press 'q' to quit.")
print("[INFO] Model expects: Bottle=0 (low prob), NotBottle=1 (high prob)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Center detection box
    h, w, _ = frame.shape
    box_size = 220
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(roi, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prob = model.predict(img, verbose=0)[0][0]

    # Bottle = LOW probability (class 0)
    is_bottle = prob < 0.5
    bottle_confidence = (1 - prob) if is_bottle else prob

    # Moving average for stability
    confidence_history.append(bottle_confidence if is_bottle else 0)
    if len(confidence_history) > HISTORY_SIZE:
        confidence_history.pop(0)
    avg_confidence = np.mean(confidence_history)

    # Visual feedback
    if is_bottle and avg_confidence > CONFIDENCE_THRESHOLD:
        color = (0, 255, 0)  # Green
        label = f"BOTTLE: {avg_confidence:.1%}"
    elif is_bottle:
        color = (0, 165, 255)  # Orange
        label = f"Maybe Bottle: {avg_confidence:.1%}"
    else:
        color = (0, 0, 255)  # Red
        label = f"NOT BOTTLE: {bottle_confidence:.1%}"

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Raw: {prob:.3f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Detection logic with moving average
    if is_bottle and avg_confidence > CONFIDENCE_THRESHOLD:
        recycle_frames += 1
        cv2.putText(frame, f"Detecting: {recycle_frames}/{FRAMES_REQUIRED}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if recycle_frames >= FRAMES_REQUIRED:
            current_time = time.time()
            if current_time - last_send_time > SEND_COOLDOWN:
                if arduino:
                    arduino.write(b"0\n")
                    print(f"[INFO] ✓✓✓ BOTTLE CONFIRMED → Arduino triggered")
                else:
                    print(f"[MOCK] ✓✓✓ BOTTLE CONFIRMED → Would open now")
                last_send_time = current_time
                recycle_frames = 0
                confidence_history.clear()
    else:
        if recycle_frames > 0:
            recycle_frames = max(0, recycle_frames - 2)  # Decay slowly

    cv2.imshow("Bottle Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()