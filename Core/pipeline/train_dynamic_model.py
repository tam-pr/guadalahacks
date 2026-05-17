import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

DATA_PATH = 'MP_Data'
ACTIONS = ['Hola, mi nombre es', 'gracias', 'perdon', 'por_favor', 'ayudame', 'Angel', 'mas', 'menos', 'parar', 'empezar', 'borrar']
NO_SEQUENCE_BURSTS = 30
SEQUENCE_LENGTH = 45
INPUT_SIZE = 126
HIDDEN_SIZE = 64
NUM_CLASSES = len(ACTIONS)
EPOCHS = 100

print("Loading sequence data from disk...")

sequences, labels = [], []
action_map = {label: num for num, label in enumerate(ACTIONS)}

for action in ACTIONS:
    for sequence in range(NO_SEQUENCE_BURSTS):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"frame_{frame_num}.npy")
            res = np.load(file_path)
            window.append(res)
        sequences.append(window)
        labels.append(action_map[action])

X = torch.tensor(np.array(sequences), dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class LsmLSTM(nn.Module):
    def __init__(self):
        super(LsmLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        return self.fc(last_out)

model = LsmLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Starting Deep Learning Training for {EPOCHS} Epochs...")

for epoch in range(EPOCHS):
    model.train()
    
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    predicted_classes = torch.argmax(test_preds, dim=1)
    correct = (predicted_classes == y_test).sum().item()
    accuracy = (correct / len(y_test)) * 100

print(f"\n--- Training Complete ---")
print(f"Final Model Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), 'dynamic_lsm_model.pth')
print("Saved model weights to 'dynamic_lsm_model.pth'")