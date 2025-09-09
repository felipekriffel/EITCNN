import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from subpastas_Dbar import dividir_e_normalizar_dados
import csv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 1, 1))

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.up1(x3)
        x5 = self.dec1(x4)
        return x5

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_path = config["train_path"]
    # val_path = config["val_path"]
    save_dir = config["save_dir"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config.get("learning_rate", 0.001)

    os.makedirs(save_dir, exist_ok=True)
    pasta_base = os.path.dirname(train_path)

    X_train, y_train, X_val, y_val = dividir_e_normalizar_dados(
        pasta_base=pasta_base, proporcao_val=0.2, normalizacao="minmax"
    )

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "unet.pth"))

    with open(os.path.join(save_dir, "losses.json"), 'w') as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=4)

    with open(os.path.join(save_dir, "losses.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])
        for i in range(epochs):
            writer.writerow([i+1, train_losses[i], val_losses[i]])

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_graph.png"))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Use: python UNET_npy_new_final.py config.json")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
