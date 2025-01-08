import ultis
import torch
from model import ChessModel
import numpy as np
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# load model for 3 

# 1 ----------
model1 = ChessModel(1).to(device)
model1.load_state_dict(torch.load("models/traindata.pth"))

X1, y1 = ultis.get_data("1")
maxso1 = abs(y1).max()

# 2 ----------
model2 = ChessModel(1).to(device)
model2.load_state_dict(torch.load("models/traindata2.pth"))

X2, y2 = ultis.get_data("2")
maxso2 = abs(y2).max()

# 3 ---------
model3 = ChessModel(1).to(device)
model3.load_state_dict(torch.load("models/traindata3.pth"))

X3, y3 = ultis.get_data("3")
maxso3 = abs(y3).max()

print(model1)

def predict(typemodel, board, device):
	board = ultis.board_to_matrix(board)
	model = model1
	maxso = maxso1

	if typemodel == 2:
		model = model2
		maxso = maxso2
	elif typemodel == 3:
		model = model3
		maxso = maxso3

	model.eval()
	with torch.no_grad():
		board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
		prediction = model(board_tensor).item()
		y = np.round((prediction-0.5) *maxso * 2) 
	return y


