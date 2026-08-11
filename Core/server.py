import cv2
import torch
import torch.nn as nn
import pickle
import numpy as np
import asyncio
import base64
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp

DYNAMIC_ACTIONS = ['Hola, mi nombre es', 'gracias', 'perdon', 'por_favor', 'ayudame', 'Angel', 'mas', 'menos', 'parar', 'empezar', 'borrar']
SEQUENCE_LENGTH = 45
INPUT_SIZE = 126
HIDDEN_SIZE = 64
NUM_CLASSES = len(DYNAMIC_ACTIONS)
WARMUP_TIME = 1.0
CONFIDENCE_SPELL = 60
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
PREDICTION_BUFFER_SIZE = 5

PER_CLASS_THRESHOLDS = {
    'Hola, mi nombre es': 88,
    'ayudame': 85,
    'gracias': 70,
    'borrar': 70,
    'parar': 70,
    'empezar': 70,
    'perdon': 72,
    'por_favor': 72,
    'Angel': 72,
    'mas': 72,
    'menos': 72,
}

with open('lsm_model.pkl', 'rb') as f:
    static_model = pickle.load(f)

class LsmLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

dynamic_model = LsmLSTM()
dynamic_model.load_state_dict(torch.load('dynamic_lsm_model.pth', map_location='cpu'))
dynamic_model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_static_features(hand_landmarks):
    raw_coords = []
    for lm in hand_landmarks.landmark:
        raw_coords.extend([lm.x, lm.y, lm.z])
    landmarks = np.array(raw_coords).reshape(21, 3)
    wrist = landmarks[0]
    centered = landmarks - wrist
    hand_size = np.linalg.norm(centered[9])
    if hand_size == 0:
        hand_size = 1
    return (centered / hand_size).flatten().reshape(1, -1)

def extract_dynamic_features(results):
    lh_data, rh_data = np.zeros(63), np.zeros(63)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            scaled = extract_static_features(hand_landmarks).flatten()
            if label == 'Left':
                lh_data = scaled
            else:
                rh_data = scaled
    return np.concatenate([lh_data, rh_data])

def capture_and_process_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None, None
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    return frame, results

def encode_frame(frame):
    small = cv2.resize(frame, (320, 240))
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode('utf-8')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    sequence = []
    prediction_history = []
    hands_present_time = 0.0
    sentence = ""
    last_word_added = ""
    last_word_time = 0.0
    current_mode = "WORDS"
    last_frame_time = time.time()
    COOLDOWN_WORDS = 1.5
    COOLDOWN_SPELL = 1.0

    try:
        while cap.isOpened():
            # ── Frame timing — match original script's rate ─────────────
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < FRAME_INTERVAL:
                await asyncio.sleep(FRAME_INTERVAL - elapsed)
            last_frame_time = time.time()

            # ── Non-blocking command check ──────────────────────────────
            try:
                command = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                if command == "TOGGLE_MODE":
                    current_mode = "SPELL" if current_mode == "WORDS" else "WORDS"
                    sequence = []
                    last_word_added = ""
                    prediction_history = []
                elif command == "CLEAR_SENTENCE":
                    sentence = ""
                    last_word_added = ""
            except (asyncio.TimeoutError, Exception):
                pass

            # ── Frame capture (direct, no executor) ────────────────────
            frame, results = capture_and_process_frame(cap)
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            current_prediction = "Waiting..."
            confidence = 0.0
            hands_visible = results.multi_hand_landmarks is not None

            # ── Reset warmup if hands disappear ────────────────────────
            if not hands_visible:
                hands_present_time = 0.0
                last_word_added = ""
                prediction_history = []
            else:
                if hands_present_time == 0.0:
                    hands_present_time = time.time()

            # ── WORDS mode ──────────────────────────────────────────────
            if current_mode == "WORDS":
                frame_features = extract_dynamic_features(results)
                sequence.append(frame_features)
                sequence = sequence[-SEQUENCE_LENGTH:]

                if not hands_visible:
                    current_prediction = "Waiting..."

                elif (time.time() - hands_present_time) < WARMUP_TIME:
                    current_prediction = "Readying..."

                elif len(sequence) == SEQUENCE_LENGTH:
                    input_tensor = torch.tensor(
                        np.array(sequence), dtype=torch.float32
                    ).unsqueeze(0)

                    with torch.no_grad():
                        preds = dynamic_model(input_tensor)
                        probs = torch.softmax(preds, dim=1)[0]
                        pred_idx = torch.argmax(probs).item()
                        confidence = probs[pred_idx].item() * 100
                        predicted_word = DYNAMIC_ACTIONS[pred_idx]

                    threshold = PER_CLASS_THRESHOLDS.get(predicted_word, 75)

                    if confidence >= threshold:
                        current_prediction = predicted_word
                        prediction_history.append(predicted_word)
                        prediction_history = prediction_history[-PREDICTION_BUFFER_SIZE:]

                        if prediction_history.count(predicted_word) >= PREDICTION_BUFFER_SIZE:
                            time_ok = (time.time() - last_word_time) > COOLDOWN_WORDS
                            word_changed = predicted_word != last_word_added

                            if word_changed and time_ok:
                                if predicted_word.lower() == 'borrar':
                                    sentence = ""
                                else:
                                    sentence += predicted_word.replace("_", " ") + " "
                                last_word_added = predicted_word
                                last_word_time = time.time()
                    else:
                        current_prediction = "Thinking..."
                        prediction_history = []

            # ── SPELL mode ──────────────────────────────────────────────
            elif current_mode == "SPELL":
                sequence = []
                if hands_visible:
                    static_features = extract_static_features(results.multi_hand_landmarks[0])
                    prediction = static_model.predict(static_features)[0]
                    confidence = np.max(static_model.predict_proba(static_features)) * 100

                    if confidence >= CONFIDENCE_SPELL:
                        current_prediction = prediction
                        time_ok = (time.time() - last_word_time) > COOLDOWN_SPELL
                        word_changed = prediction != last_word_added

                        if word_changed and time_ok:
                            if prediction.lower() == 'borrar':
                                sentence = sentence[:-1]
                            elif prediction.lower() == 'espacio':
                                sentence += " "
                            else:
                                sentence += prediction
                            last_word_added = prediction
                            last_word_time = time.time()
                    else:
                        current_prediction = "Reading..."

            # ── Encode and send ─────────────────────────────────────────
            cv2.putText(
                frame, f"Mode: {current_mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2
            )
            b64_image = encode_frame(frame)

            await websocket.send_json({
                "prediction": current_prediction,
                "confidence": float(confidence),
                "sentence": sentence,
                "mode": current_mode,
                "image": b64_image
            })

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()