import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

csv_file = 'lsm_raw_data.csv'
print("Loading dataset...")
df = pd.read_csv(csv_file)

X = df.drop(columns=['label']).values
y = df['label'].values

print("Normalizing hand geometry...")
X_normalized = []

for row in X:
    landmarks = row.reshape(21, 3)
    wrist = landmarks[0]
    
    centered_landmarks = landmarks - wrist
    
    hand_size = np.linalg.norm(centered_landmarks[9])
    if hand_size == 0: hand_size = 1
    scaled_landmarks = centered_landmarks / hand_size
    
    X_normalized.append(scaled_landmarks.flatten())

X_normalized = np.array(X_normalized)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

print("Training local AI model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Training Complete ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

model_filename = 'lsm_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Saved trained model asset to: '{model_filename}'")