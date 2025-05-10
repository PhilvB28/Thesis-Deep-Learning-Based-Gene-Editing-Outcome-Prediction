import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.stats import pearsonr, spearmanr

from utils.data_utils import CRISPRDataset_557, CRISPRDataset_6, CRISPRDataset_noframeshift, CRISPRDataset_tokenized, CRISPRDataset_DeepIndel


def test(test_dfs, model, model_save_file, device, batch_size=32, table=False):
    """
    Returns:
        If table is False:
            A dictionary with overall and per-label MSE, Pearson, Spearman, and AUC values
            for each test dataset, or None if there was an error loading the model.
        If table is True:
            A tuple (results, csv_table) where csv_table is a CSV-formatted string that can be
            copied into an Excel sheet.
    """
    forecast_lindel_test_df = test_dfs[0]
    forecast_test_df = test_dfs[1]
    lindel_test_df = test_dfs[2]
    sprout_test_df = test_dfs[3]

    # Move model to device.
    model = model.to(device)

    # Attempt to load model weights.
    try:
        model.load_state_dict(torch.load(model_save_file, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set model to evaluation mode.
    model.eval()

    test_datasets = {
        "forecast_lindel": forecast_lindel_test_df,
        "forecast": forecast_test_df,
        "lindel": lindel_test_df,
        "sprout": sprout_test_df,
    }
    results = {}
    # List to collect rows for the CSV table.
    table_rows = []

    datasets = list(test_datasets.items())
    for idx, (name, ds) in enumerate(datasets):
        # Initialize dataset.
        test_dataset = CRISPRDataset_DeepIndel(ds)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        # Concatenate predictions and targets.
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Overall metrics for continuous outputs.
        mse = mean_squared_error(all_targets, all_preds)
        pearson_corr, _ = pearsonr(all_targets.flatten(), all_preds.flatten())
        spearman_corr, _ = spearmanr(all_targets.flatten(), all_preds.flatten())

        # Compute AUC for each label.
        num_labels = all_preds.shape[1]
        aucs = []
        per_label_results = {}
        for label in range(num_labels):
            # Per-label continuous metrics.
            mse_label = mean_squared_error(all_targets[:, label], all_preds[:, label])
            pearson_label, _ = pearsonr(all_targets[:, label], all_preds[:, label])
            spearman_label, _ = spearmanr(all_targets[:, label], all_preds[:, label])

            # Threshold the true values at the median to get binary labels.
            median_val = np.median(all_targets[:, label])
            y_true_binary = (all_targets[:, label] >= median_val).astype(int)
            # Use the predicted continuous values as scores.
            try:
                auc_label = roc_auc_score(y_true_binary, all_preds[:, label])
            except ValueError:
                auc_label = np.nan  # In case only one class is present
            aucs.append(auc_label)

            per_label_results[f"label_{label}"] = {
                'MSE': mse_label,
                'Pearson': pearson_label,
                'Spearman': spearman_label,
                'AUC': auc_label
            }

        # Overall AUC is computed as the average of the per-label AUCs (ignoring nan values).
        auc_overall = np.nanmean(aucs)

        results[name] = {
            'overall': {
                'MSE': mse,
                'Pearson': pearson_corr,
                'Spearman': spearman_corr,
                'AUC': auc_overall
            },
            'per_label': per_label_results,
        }

        # Append overall metrics to table_rows.
        table_rows.append({
            "MSE": f"{mse:.4f}",
            "Pearson": f"{pearson_corr:.4f}",
            "Spearman": f"{spearman_corr:.4f}",
            "AUC": f"{auc_overall:.4f}"
        })

        # Append per-label metrics.
        for label, metrics in per_label_results.items():
            auc_str = f"{metrics['AUC']:.4f}" if not np.isnan(metrics['AUC']) else "nan"
            table_rows.append({
                "MSE": f"{metrics['MSE']:.4f}",
                "Pearson": f"{metrics['Pearson']:.4f}",
                "Spearman": f"{metrics['Spearman']:.4f}",
                "AUC": auc_str
            })

        # Add an empty row between different datasets (except after the last one).
        if idx < len(datasets) - 1:
            table_rows.append({
                "MSE": "",
                "Pearson": "",
                "Spearman": "",
                "AUC": ""
            })

        # Print results for this dataset.
        print(f"Results for dataset '{name}':")
        print(f"  Overall MSE: {mse:.4f}")
        print(f"  Overall Pearson Correlation: {pearson_corr:.4f}")
        print(f"  Overall Spearman Correlation: {spearman_corr:.4f}")
        print(f"  Overall AUC: {auc_overall:.4f}")
        print("  Per-label metrics:")
        for label, metrics in per_label_results.items():
            auc_str = f"{metrics['AUC']:.4f}" if not np.isnan(metrics['AUC']) else "nan"
            print(f"    {label}: MSE: {metrics['MSE']:.4f}, "
                  f"Pearson: {metrics['Pearson']:.4f}, "
                  f"Spearman: {metrics['Spearman']:.4f}, "
                  f"AUC: {auc_str}")
        print("-" * 40)

    # If table is True, create a CSV-formatted string from table_rows.
    if table:
        df_table = pd.DataFrame(table_rows)
        csv_table = df_table.to_csv(index=False)
        print(csv_table)
        return results, csv_table

    return results

def calculate_auc(y_true, y_pred):
   Y_test_median = np.median(y_true)
   Y_test_binary = np.where(y_true>=Y_test_median,1.0,y_true)
   Y_test_binary = np.where(y_true<Y_test_median,0.0,y_true)
   auc = roc_auc_score(Y_test_binary,y_pred)
   return auc