import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pickle
import time

DYNAMIC_ACTIONS = ['Hola, mi nombre es', 'gracias', 'perdon', 'por_favor', 'ayudame', 'Angel', 'mas', 'menos', 'parar', 'empezar', 'borrar']
SEQUENCE_LENGTH = 45
INPUT_SIZE = 126
HIDDEN_SIZE = 64
NUM_CLASSES = len(DYNAMIC_ACTIONS)

print("Loading Tier 1 Brain (Static Alphabet)...")
with open('lsm_model.pkl', 'rb') as f:
    static_model = pickle.load(f)

print("Loading Tier 2 Brain (Dynamic Words)...")
class LsmLSTM(nn.Module):
    def __init__(self):
        super(LsmLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        return self.fc(last_out)

dynamic_model = LsmLSTM()
dynamic_model.load_state_dict(torch.load('dynamic_lsm_model.pth'))
dynamic_model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

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

cap = cv2.VideoCapture(0)
sequence = [] 

current_mode = "WORDS"
current_prediction = "Waiting..."
confidence = 0.0
sentence = ""           
last_word_added = ""    
last_word_time = 0.0    
COOLDOWN_WORDS = 1.5 
COOLDOWN_SPELL = 1.0  
hands_present_time = 0.0  
WARMUP_TIME = 1.0         

print("\n=== SYSTEM ONLINE ===")
print("[M] Toggle Mode (Words <-> Spell) | [Q] Quit\n")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # Note: Skeleton drawing logic completely removed for clean presentation UI!

    if current_mode == "WORDS":
        frame_features = extract_dynamic_features(results)
        sequence.append(frame_features)
        sequence = sequence[-SEQUENCE_LENGTH:]

        hands_visible = np.sum(frame_features) != 0.0

        if not hands_visible:
            current_prediction = "Waiting..."
            confidence, last_word_added = 0.0, ""
            hands_present_time = 0.0  
        else:
            if hands_present_time == 0.0:
                hands_present_time = time.time()

            if (time.time() - hands_present_time) < WARMUP_TIME:
                current_prediction = "Readying..."
                confidence = 0.0
                
            elif len(sequence) == SEQUENCE_LENGTH:
                input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    preds = dynamic_model(input_tensor)
                    probs = torch.softmax(preds, dim=1)[0] 
                    pred_idx = torch.argmax(probs).item()
                    confidence = probs[pred_idx].item() * 100
                    
                    if confidence >= 75:
                        current_prediction = DYNAMIC_ACTIONS[pred_idx]
                        
                        if current_prediction != last_word_added and (time.time() - last_word_time) > COOLDOWN_WORDS:
                            if current_prediction == 'borrar':
                                sentence = "" 
                            else:
                                sentence += current_prediction.replace("_", " ") + " "
                                
                            last_word_added, last_word_time = current_prediction, time.time()
                    else:
                        current_prediction = "Transitioning..."

    elif current_mode == "SPELL":
        sequence = [] 
        if results.multi_hand_landmarks:
            primary_hand = results.multi_hand_landmarks[0]
            input_vector = extract_static_features(primary_hand)
            
            prediction = static_model.predict(input_vector)[0]
            probabilities = static_model.predict_proba(input_vector)[0]
            confidence = np.max(probabilities) * 100
            
            if confidence >= 75:
                current_prediction = prediction
                
                if current_prediction != last_word_added and (time.time() - last_word_time) > COOLDOWN_SPELL:
                    if current_prediction == 'BORRAR': 
                        sentence = sentence[:-1] 
                    elif current_prediction == 'ESPACIO': 
                        sentence += " "
                    else:
                        sentence += current_prediction 
                        
                    last_word_added, last_word_time = current_prediction, time.time()
            else:
                current_prediction = "Reading..."
        else:
            current_prediction = "Waiting..."
            confidence, last_word_added = 0.0, ""


    # Bottom Sentence Banner
    cv2.rectangle(image, (0, 420), (640, 480), (0, 0, 0), -1)
    
    # Subtle Mode Indicator in the bottom right corner
    mode_color = (245, 117, 16) if current_mode == "WORDS" else (16, 117, 245)
    cv2.putText(image, f"MODE: {current_mode}", (480, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)

    # Main Subtitle Text
    display_sentence = sentence if len(sentence) < 35 else "..." + sentence[-30:]
    cv2.putText(image, display_sentence, (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('LSM AI Translation', image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):  
        current_mode = "SPELL" if current_mode == "WORDS" else "WORDS"
        last_word_added = "" 
        sequence = []        

cap.release()
cv2.destroyAllWindows()