import torch
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

from train import train_valid
from test import test
from trainsoft import train_valid_soft

from models.CRISPR_Caps_hard import CrisprCaps_hard
from models.CRISPR_Caps_soft import CrisprCaps_soft


seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If GPU usage
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    set_seed(seed)
    print(f'Using device: {device}')

    # Select model
    model = CrisprCaps_hard(input_channels=4, sequence_length=60, in_dim=16, out_dim=16, num_routing=2)

    # Select model params
    params ="./saved_models/CapsNet_hard.pth"

    # Paths
    forecast_path = './datasets/Forecast K562 (n=35129).csv'
    lindel_path = './datasets/Lindel HEK293t (n=4591).csv'
    sprout_path = './datasets/Sprout Tcell (n=1603).csv'

    forecast_df = pd.read_csv(forecast_path)
    lindel_df = pd.read_csv(lindel_path)
    sprout_df = pd.read_csv(sprout_path)
    forecast_lindel_df = pd.concat([forecast_df, lindel_df], axis=0)


    # Data
    forecast_lindel_train_df, temp_df = train_test_split(forecast_lindel_df, test_size=0.2, random_state=42)
    forecast_train_df, temp2_df = train_test_split(forecast_df, test_size=0.2, random_state=42)
    lindel_train_df, temp3_df = train_test_split(lindel_df, test_size=0.2, random_state=42)

    # validation and test sets (10% each of the original data)
    forecast_lindel_valid_df, forecast_lindel_test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    forecast_valid_df, forecast_test_df = train_test_split(temp2_df, test_size=0.5, random_state=42)
    lindel_valid_df, lindel_test_df = train_test_split(temp3_df, test_size=0.5, random_state=42)

    # Run training
    print('Initializing Training...')

    train_valid(
        train_df=lindel_train_df,
        valid_df=lindel_valid_df,
        model_save_file=params,
        model=model,
        epochs=150,
        batch_size=32,
        lr=0.001,
        patience=10
    )

    # Run testing
    print('Initializing Testing...')
    print('Testing' + params)

    test([forecast_lindel_test_df, forecast_test_df, lindel_test_df, sprout_df], model, params, device, batch_size=32, table=True)
