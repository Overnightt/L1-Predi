import torch
import torch.nn as nn
import torch.optim as optim
from data_check import check_data
from model import L1_Predictor
from dataset_builder import build_dataset

#This is where we train the model

n_matches=15
n_features=40
model = L1_Predictor(n_matches,n_features)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
matches=check_data("../data/season-2425.csv")
X, Y = build_dataset(matches,n_matches)

epochs=250

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_function(outputs, Y.squeeze())
    loss.backward()
    optimizer.step()
    preds = torch.argmax(outputs, dim=1)
    accuracy = (preds == Y.squeeze()).float().mean().item()
    print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}")
torch.save(model.state_dict(),"L1_predictor_v1.pth")