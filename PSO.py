import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyswarms as ps
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.CRISPR_Caps_soft import CapsNetRegressorSoftSharing_3layer
from models.CRISPR_Caps_hard import CapsNetRegressorHardSharing_3_layer

from utils.data_utils import CRISPRDataset_DeepIndel

forecast_path = './datasets/Forecast K562 (n=35129).csv'
lindel_path = './datasets/Lindel HEK293t (n=4591).csv'
sprout_path = './datasets/Sprout Tcell (n=1603).csv'

forecast_df = pd.read_csv(forecast_path)
lindel_df = pd.read_csv(lindel_path)
sprout_df = pd.read_csv(sprout_path)

halfdata_df, _ = train_test_split(forecast_df, test_size=0.5, random_state=42)
train_df, df_valid = train_test_split(halfdata_df, test_size=0.1, random_state=42)

# Define your function to train and evaluate the Capsule Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Function to train and evaluate the model
def objective_function(hyperparams):
    """
    PSO Objective function that builds, trains, and evaluates the Capsule Network using CRISPRDataset_noframeshift.
    """
    all_mse = []  # Store MSE for each particle

    # Loop through each particle (set of hyperparameters)
    for i, params in enumerate(hyperparams):
        print(f"Processing particle {i + 1}/{len(hyperparams)}")
        log_lr, in_dim, out_dim, num_routing, batch_size = params

        #learning_rate = max(1e-6, min(1e-3, learning_rate))

        log_lr = max(-6, min(-3, log_lr))
        learning_rate = 10 ** log_lr
        in_dim = int(round(max(4, min(32, out_dim))))
        out_dim = int(round(max(4, min(32, out_dim))))
        num_routing = int(round(max(1, min(5, num_routing))))
        batch_size = int(round(max(16, min(128, batch_size))))

        model = CapsNetRegressorHardSharing_3_layer(
            input_channels=4,
            sequence_length=60,
            in_dim=in_dim,
            out_dim=out_dim,
            num_routing=num_routing,
        ).to(device)

        # Loss and optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss() #MSE Huber
        lambda_reg = 1e-4

        # Load datasets using CRISPRDataset
        train_dataset = CRISPRDataset_DeepIndel(train_df, device=device)
        valid_dataset = CRISPRDataset_DeepIndel(df_valid, device=device)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        model.train()
        for epoch in range(3):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                #Comment out for hardsharing:
                #main_loss = criterion(outputs, batch_y)
                # compute soft‐sharing reg loss (L2 by default)
                #reg_loss = model.compute_soft_sharing_loss(p=2)
                #loss = main_loss + lambda_reg * reg_loss

                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        y_true, y_pred_all = [], []

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                preds = model(batch_x).cpu().numpy()
                y_true.append(batch_y.cpu().numpy())
                y_pred_all.append(preds)

        # Compute MSE
        y_true = np.vstack(y_true)
        y_pred_all = np.vstack(y_pred_all)
        mse = mean_squared_error(y_true, y_pred_all)

        all_mse.append(mse)

    return np.array(all_mse)



# Define hyperparameter search space
bounds = np.array([
    [-5, -3],        # Learning Rate (log)
    [4, 32],         # Capsule Input Dimension (in_dim)
    [4, 32],         # Capsule Output Dimension (out_dim)
    [1, 4],          # Number of Routing Iterations
    [16, 128],       # Batch Size
])

# PSO optimizer settings
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.8}

optimizer = ps.single.GlobalBestPSO(
    n_particles=10,  # Number of candidate solutions
    dimensions=5,    # Number of hyperparameters to optimize
    options=options,
    bounds=bounds.T
)

# Run optimization
best_score, best_params = optimizer.optimize(objective_function, iters=15)
print("Best Hyperparameters:", best_params)
print("Best MSE Score:", best_score)