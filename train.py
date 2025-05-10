#This File contains the training function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

from utils.data_utils import CRISPRDataset_557, CRISPRDataset_6, CRISPRDataset_noframeshift, CRISPRDataset_tokenized, CRISPRDataset_DeepIndel


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_valid(train_df, valid_df, model_save_file, model, epochs, batch_size, lr, patience):

    # Tokenized Dataset
    train_dataset = CRISPRDataset_DeepIndel(train_df)
    valid_dataset = CRISPRDataset_DeepIndel(valid_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Move model to GPU
    model.to(device)

    # Prepare the model, loss function, and optimizer
    criterion = nn.BCELoss()  #HuberLoss #MSELoss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize for early stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_valid_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            #print("Debugging Info:")
            #print("Model outputs:", outputs)
            #print("True values:", y_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            #print("conv1 gradients NaN?", torch.isnan(model.conv1.weight.grad).any())
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item() * X_batch.size(0)
        valid_loss /= len(valid_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.10f}, Valid Loss: {valid_loss:.10f}')

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the model
    torch.save(model.state_dict(), model_save_file)
    print("Training abgeschlossen und Modell gespeichert.")

