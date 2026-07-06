import torch
import torch.nn as nn
import torch.optim as optim
from data_check import check_data
from model import L1_Predictor
from dataset_builder import build_dataset

#This is where we train the model

n_matches=26
n_features=40
model = L1_Predictor(n_features)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.00077)
matches=check_data("../data/seasons-17-26.csv")
split_idx = int(len(matches) * 0.8)
train_matches = matches.iloc[:split_idx]
val_matches   = matches.iloc[split_idx:]
X_Train, Y_Train = build_dataset(train_matches,n_matches)
X_val, Y_val = build_dataset(val_matches,n_matches)
previous_acc = float('-inf')
previous_loss = float("inf")
epochs=150

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_Train)
    loss = loss_function(outputs, Y_Train.squeeze())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_function(val_outputs, Y_val.squeeze())
        val_preds = torch.argmax(val_outputs, dim=1)
        val_acc = (val_preds == Y_val.squeeze()).float().mean().item()
        if previous_acc<val_acc  or (val_acc == previous_acc and val_loss < previous_loss):
            print(f"i save the model with acc {val_acc}")
            torch.save(model.state_dict(),"../Model/YOUR_MODEL_NAME.pth")
            previous_acc = val_acc
            previous_loss = val_loss
    print(
        f"Epoch {epoch} | "
        f"Train loss: {loss.item():.3f} | "
        f"Val loss: {val_loss.item():.3f} | "
        f"Val acc: {val_acc:.3f}"
    )


