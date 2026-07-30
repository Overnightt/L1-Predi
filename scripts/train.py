import torch
import torch.nn as nn
import torch.optim as optim
from data_check import check_data
from model import FC_Predictor
from dataset_builder import build_dataset
from normalisation_stats import norm_stats

torch.manual_seed(42)

#This is where we train the model

n_matches=26
n_features=40
static_size=5
model = FC_Predictor(n_features,static_size)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.007) # MODIFY learning rate here

matches=check_data("../data/seasons-19-26.csv")
squad_values = check_data("../data/teams-19-26.csv", date_column="SeasonStart")

split_idx = int(len(matches) * 0.8)
train_matches = matches.iloc[:split_idx]
val_matches   = matches.iloc[split_idx:]

stats = norm_stats(train_matches, squad_values)
X_Train, H_Train, Y_Train = build_dataset(train_matches,n_matches,stats)
X_val, H_val ,Y_val = build_dataset(val_matches,n_matches,stats)

previous_acc = float('-inf')
previous_loss = float("inf")

epochs=150 # MODIFY epochs here

# The training loop
for epoch in range(epochs):

    model.train()
    optimizer.zero_grad()
    outputs = model(X_Train,H_Train)
    loss = loss_function(outputs, Y_Train.squeeze())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val,H_val)
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


