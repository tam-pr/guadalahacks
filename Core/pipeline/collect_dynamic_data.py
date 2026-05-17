import cv2
import mediapipe as mp
import numpy as np
import os
import time

ACTIONS = np.array(['borrar'])
NO_SEQUENCE_BURSTS = 30                 
SEQUENCE_LENGTH = 45                  
DATA_PATH = os.path.join('MP_Data') 

for action in ACTIONS: 
    for sequence in range(NO_SEQUENCE_BURSTS):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7  
)

def extract_and_normalize_landmarks(results):
    lh_data = np.zeros(63)
    rh_data = np.zeros(63)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            
            raw_coords = []
            for landmark in hand_landmarks.landmark:
                raw_coords.extend([landmark.x, landmark.y, landmark.z])
                
            landmarks = np.array(raw_coords).reshape(21, 3)
            wrist = landmarks[0]
            centered = landmarks - wrist
            hand_size = np.linalg.norm(centered[9])
            if hand_size == 0: hand_size = 1
            scaled = (centered / hand_size).flatten()
            
            if hand_label == 'Left':
                lh_data = scaled
            else:
                rh_data = scaled
                
    return np.concatenate([lh_data, rh_data])

cap = cv2.VideoCapture(0)

print("=== 2-Hand Dynamic Sequence Collector Initialized ===")
print("Press 'S' to start recording. Press 'Q' to quit.\n")

for action in ACTIONS:
    for sequence in range(NO_SEQUENCE_BURSTS):
        frame_num = 0
        recording = False
        sequence_data = []

        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if recording:
                frame_features = extract_and_normalize_landmarks(results)
                sequence_data.append(frame_features)
                frame_num += 1

            if not recording:
                cv2.putText(image, f"READY: Record '{action}' | Burst #{sequence}", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.circle(image, (30, 35), 10, (0, 0, 255), -1)
                cv2.putText(image, f"RECORDING {frame_num}/{SEQUENCE_LENGTH}", (55, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('LSM Dynamic Collector', image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and not recording:
                recording = True
                frame_num = 0
                sequence_data = []
                time.sleep(0.3) 

            if frame_num == SEQUENCE_LENGTH:
                res_path = os.path.join(DATA_PATH, action, str(sequence))
                for f_idx, frame_arr in enumerate(sequence_data):
                    np.save(os.path.join(res_path, f"frame_{f_idx}.npy"), frame_arr)
                print(f"Successfully saved burst #{sequence} for '{action}'")
                break

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()
print("\nAll dynamic data successfully collected!")