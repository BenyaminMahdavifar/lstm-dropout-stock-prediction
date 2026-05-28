# experiments/run_all_experiments.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to sys.path so we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import load_and_scale_data, create_sequences, prepare_test_data
from src.models import get_all_models
from src.evaluation import evaluate_all_metrics

def main():
    # ------------------------------
    # 1. Load & preprocess data
    # ------------------------------
    print("Loading and preprocessing data...")
    train_scaled, test_values, scaler = load_and_scale_data(
        train_path='data/train.csv',
        test_path='data/test.csv',
        feature_col='Open'
    )
    X_train, y_train = create_sequences(train_scaled, timesteps=60)
    X_test = prepare_test_data(train_scaled, test_values, scaler, timesteps=60)
    input_shape = (X_train.shape[1], 1)
    y_true = test_values  # shape (n_samples, 1)

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # ------------------------------
    # 2. Build & train all models
    # ------------------------------
    models = get_all_models(input_shape)
    predictions = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        predictions[name] = pred

    # ------------------------------
    # 3. Evaluate all models
    # ------------------------------
    print("\nEvaluating models...")
    results = []
    for name, pred in predictions.items():
        rmse, mae, mape_val, da = evaluate_all_metrics(y_true, pred)
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape_val,
            'Directional Accuracy (%)': da
        })
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape_val:.2f}%, DA={da:.2f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/metrics_summary.csv', index=False)
    print("\nMetrics saved to results/metrics_summary.csv")

    # ------------------------------
    # 4. Plot & save figures
    # ------------------------------
    os.makedirs('results', exist_ok=True)

    # Figure 1: Comparison curves
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, color='black', linewidth=2, label='Real Price')
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for (name, pred), color in zip(predictions.items(), colors):
        plt.plot(pred, color=color, linestyle='--', label=name)
    plt.title('Comparison of Dropout Patterns – Google Stock Price Prediction')
    plt.xlabel('Time (test samples)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison_curves.png")

    # Figure 2: Metrics bar chart
    models_names = results_df['Model']
    x = np.arange(len(models_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - width/2, results_df['RMSE'], width, label='RMSE', color='skyblue')
    ax1.bar(x + width/2, results_df['MAE'], width, label='MAE', color='lightcoral')
    ax1.set_ylabel('Error Value')
    ax1.set_title('RMSE and MAE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.bar(x - width/2, results_df['MAPE (%)'], width, label='MAPE (%)', color='lightgreen')
    ax2.bar(x + width/2, results_df['Directional Accuracy (%)'], width, label='Directional Accuracy (%)', color='gold')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('MAPE and Directional Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/metrics_barchart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved metrics_barchart.png")

    # Figure 3: Residuals for the best model (ascending)
    best_name = 'Ascending (0.2->0.5)'
    if best_name in predictions:
        best_pred = predictions[best_name]
        residuals = y_true.flatten() - best_pred.flatten()

        plt.figure(figsize=(14, 4))
        plt.plot(residuals, color='purple', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', label='Zero error')
        plt.title(f'Residuals over Time - {best_name}')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Prediction Error')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig('results/residuals_time.png', dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of Residuals - {best_name}')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('results/residuals_hist.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved residuals plots.")

    print("\nAll experiments completed successfully!")

if __name__ == '__main__':
    main()