import os
import numpy as np # type: ignore
import time
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from ultis import  create_input_for_nn, encode_moves, get_data
from dataset import ChessDataset
from model import ChessModel
from tqdm import tqdm

print("---------get data set------------")
X, y = create_input_for_nn(50000)
# X, y = get_data()
# maxso = abs(y).max()
# print(maxso)
# print(y)
y = np.asarray(y / abs(y).max() / 2 + 0.5, dtype=np.float32)
# print(y)

# y = np.asarray(np.round((y-0.5) *maxso * 2) , dtype=np.float32)
# print(y)

print("-------------prepare training---------")
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
print(len(y))

dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Initialization
model = ChessModel(1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


# Training

def predict(model, board, device):
	model.eval()
	with torch.no_grad():
		board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
		prediction = model(board_tensor).item()
	return prediction

def calculate_r2(model, dataloader, device):
	model.eval()  # Set model to evaluation mode
	total_ssr = 0.0  # Sum of squared residuals
	total_sst = 0.0  # Total sum of squares
	y_mean = torch.tensor(y.mean(), device=device)  # Mean of the target values

	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs).squeeze(1)
			total_ssr += ((labels - outputs) ** 2).sum().item()
			total_sst += ((labels - y_mean) ** 2).sum().item()

	r2 = 1 - (total_ssr / total_sst)
	print(f"R² Accuracy Score: {r2:.4f}")
	return r2


print("-----------now train-----------")
num_epochs = 100
best_loss = float('inf')
patience = 5  # Stop if no improvement for 5 epochs
no_improvement = 0
for epoch in range(num_epochs):
	start_time = time.time()
	model.train()
	running_loss = 0.0

	for inputs, labels in tqdm(dataloader):
		inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
		optimizer.zero_grad()

		outputs = model(inputs).squeeze(1)  # Raw logits

		# Compute loss
		loss = criterion(outputs, labels)
		loss.backward()
		
		# Gradient clipping
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		
		optimizer.step()
		running_loss += loss.item()

	end_time = time.time()
	epoch_time = end_time - start_time
	minutes: int = int(epoch_time // 60)
	seconds: int = int(epoch_time) - minutes * 60
	print(f'Epoch {epoch}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}')
	if running_loss < best_loss:
		best_loss = running_loss
		torch.save(model.state_dict(), "models/traindata3.pth")
		no_improvement = 0
	else:
		no_improvement += 1

	if no_improvement >= patience:
		print("Early stopping triggered!")
		break

print("save model")
print("--------test data---------")
calculate_r2(model, dataloader, device)

i = 0
for g in  X:
	print(predict(model, g, device), y[i])
	i+=1





