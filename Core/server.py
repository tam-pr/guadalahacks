import cv2
import torch
import torch.nn as nn
import pickle
import numpy as np
import asyncio
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp

# Constants and model configuration
DYNAMIC_ACTIONS = ['Hola, mi nombre es', 'gracias', 'perdon', 'por_favor', 'ayudame', 'Angel', 'mas', 'menos', 'parar', 'empezar', 'borrar']
SEQUENCE_LENGTH = 45
INPUT_SIZE = 126
HIDDEN_SIZE = 64
NUM_CLASSES = len(DYNAMIC_ACTIONS)

# Load static model
with open('lsm_model.pkl', 'rb') as f:
    static_model = pickle.load(f)

# Define and load dynamic model
class LsmLSTM(nn.Module):
    def __init__(self):
        super(LsmLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

dynamic_model = LsmLSTM()
dynamic_model.load_state_dict(torch.load('dynamic_lsm_model.pth', map_location=torch.device('cpu')))
dynamic_model.eval()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Feature extraction helpers
def extract_static_features(hand_landmarks):
    raw_coords = []
    for landmark in hand_landmarks.landmark:
        raw_coords.extend([landmark.x, landmark.y, landmark.z])
    landmarks = np.array(raw_coords).reshape(21, 3)
    wrist = landmarks[0]
    centered = landmarks - wrist
    hand_size = np.linalg.norm(centered[9])
    if hand_size == 0: hand_size = 1
    return (centered / hand_size).flatten().reshape(1, -1)

def extract_dynamic_features(results):
    lh_data, rh_data = np.zeros(63), np.zeros(63)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            scaled = extract_static_features(hand_landmarks).flatten()
            if hand_label == 'Left': lh_data = scaled
            else: rh_data = scaled
    return np.concatenate([lh_data, rh_data])

# App setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Websocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    sequence = []
    sentence = ""
    last_word_added = ""
    current_mode = "WORDS"
    missing_frames = 0
    
    try:
        while cap.isOpened():
            # Handle UI commands
            try:
                command = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if command == "TOGGLE_MODE":
                    current_mode = "SPELL" if current_mode == "WORDS" else "WORDS"
                    sequence = []
                elif command == "CLEAR_SENTENCE":
                    sentence = ""
                    last_word_added = ""
            except asyncio.TimeoutError:
                pass 
            
            # Capture and process frame
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            
            current_prediction = "Waiting..."
            confidence = 0.0
            hands_visible = results.multi_hand_landmarks is not None
            
            frame_features = extract_dynamic_features(results)
            sequence.append(frame_features)
            sequence = sequence[-SEQUENCE_LENGTH:]
            
            # Cooldown logic
            cooldown_limit = 15 if current_mode == "WORDS" else 2

            if not hands_visible:
                missing_frames += 1
                if missing_frames > cooldown_limit: 
                    last_word_added = "" 
            else:
                missing_frames = 0 
                
                # Process WORDS mode
                if current_mode == "WORDS":
                    dynamic_clear_triggered = False
                    
                    if len(sequence) == SEQUENCE_LENGTH:
                        input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            preds = dynamic_model(input_tensor)
                            probs = torch.softmax(preds, dim=1)[0]
                            pred_idx = torch.argmax(probs).item()
                            dyn_confidence = probs[pred_idx].item() * 100
                            
                            # Global erase
                            if dyn_confidence >= 80 and DYNAMIC_ACTIONS[pred_idx].lower() == 'borrar':
                                if DYNAMIC_ACTIONS[pred_idx] != last_word_added:
                                    sentence = ""
                                    last_word_added = DYNAMIC_ACTIONS[pred_idx]
                                current_prediction = "BORRAR (ALL)"
                                confidence = dyn_confidence
                                dynamic_clear_triggered = True

                    if not dynamic_clear_triggered:
                        if len(sequence) == SEQUENCE_LENGTH:
                            confidence = dyn_confidence
                            if confidence >= 75:
                                current_prediction = DYNAMIC_ACTIONS[pred_idx]
                                if current_prediction != last_word_added:
                                    if current_prediction.lower() == 'borrar':
                                        sentence = ""
                                    else:
                                        sentence += current_prediction.replace("_", " ") + " "
                                    last_word_added = current_prediction
                            else:
                                current_prediction = "Thinking..."
                        else:
                            current_prediction = "Waiting..."
                                    
                # Process SPELL mode
                elif current_mode == "SPELL":
                    static_features = extract_static_features(results.multi_hand_landmarks[0])
                    prediction = static_model.predict(static_features)[0]
                    prob = np.max(static_model.predict_proba(static_features)) * 100
                    confidence = prob
                    
                    if confidence >= 60:
                        current_prediction = prediction
                        if current_prediction != last_word_added:
                            if current_prediction.lower() == 'borrar':
                                sentence = sentence[:-1] 
                            elif current_prediction.lower() == 'espacio':
                                sentence += " "
                            else:
                                sentence += current_prediction
                            last_word_added = current_prediction

            # Render and encode image
            cv2.putText(frame, f"AI Mode: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            b64_image = base64.b64encode(buffer).decode('utf-8')

            # Send payload to React
            await websocket.send_json({
                "prediction": current_prediction,
                "confidence": float(confidence),
                "sentence": sentence,
                "mode": current_mode,
                "image": b64_image
            })
            
            await asyncio.sleep(0.001) 
            
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()