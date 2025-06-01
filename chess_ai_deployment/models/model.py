import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessNet(nn.Module):
    """Neural network for chess position evaluation."""
    
    def __init__(self, input_size=776):
        super(ChessNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def load_data():
    data_dir = "d:/envy/chess-ml/env/data/processed"
    positions_path = os.path.join(data_dir, "chess_data_large_positions.npy")
    outcomes_path = os.path.join(data_dir, "chess_data_large_outcomes.npy")
    
    if not os.path.exists(positions_path) or not os.path.exists(outcomes_path):
        logger.error("Processed data files not found. Run data_prep.py first.")
        return None, None
    
    positions = np.load(positions_path)
    outcomes = np.load(outcomes_path)
    
    logger.info(f"Loaded {len(positions)} positions and {len(outcomes)} outcomes")
    
    return positions, outcomes

def prepare_data_loaders(positions, outcomes, batch_size=64, train_ratio=0.8):
    X = torch.tensor(positions, dtype=torch.float32)
    y = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1)
    
    dataset_size = len(positions)
    train_size = int(dataset_size * train_ratio)
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    logger.info(f"Created data loaders with {train_size} training samples and {dataset_size - train_size} validation samples")
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training on {device}")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_path = "d:/envy/chess-ml/env/models/best_chess_model.pt"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model with validation loss: {val_loss:.6f}")
    
    model.load_state_dict(torch.load(best_model_path))
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    return model

def save_model(model, filename="chess_model.pt"):
    model_dir = "d:/envy/chess-ml/env/models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, filename)
    torch.save(model.state_dict(), model_path)
    
    model.eval()
    example_input = torch.zeros(1, 776, dtype=torch.float32)
    traced_model = torch.jit.trace(model, example_input)
    traced_model_path = os.path.join(model_dir, f"traced_{filename}")
    torch.jit.save(traced_model, traced_model_path)
    
    logger.info(f"Model saved to {model_path} and {traced_model_path}")

def main():
    logger.info("Starting model training")
    
    positions, outcomes = load_data()
    if positions is None or outcomes is None:
        return
    
    train_loader, val_loader = prepare_data_loaders(positions, outcomes)
    
    model = ChessNet()
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    trained_model = train_model(model, train_loader, val_loader)
    save_model(trained_model)
    
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    main()
