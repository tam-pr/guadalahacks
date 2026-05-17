import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

csv_file = 'lsm_raw_data.csv'

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(headers)

current_label = input("Enter the LSM letter you are about to record (e.g., 'A', 'B'): ").upper()

cap = cv2.VideoCapture(0)
print(f"Recording data for '{current_label}'. Press 'S' to save a frame. Press 'Q' to quit.")
count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                row = [current_label]
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                count += 1
                print(f"Saved array #{count} for '{current_label}'")

    cv2.imshow('LSM Data Collector', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()