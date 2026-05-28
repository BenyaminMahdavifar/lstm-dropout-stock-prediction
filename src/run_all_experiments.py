import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessing import load_and_scale_data, create_sequences, prepare_test_data
from src.models import get_all_models
from src.evaluation import evaluate_all_metrics

def main():
    # 1. Load & preprocess
    train_scaled, test_values, scaler = load_and_scale_data()
    X_train, y_train = create_sequences(train_scaled)
    X_test = prepare_test_data(train_scaled, test_values, scaler)
    input_shape = (X_train.shape[1], 1)
    
    # 2. Build & train models
    models = get_all_models(input_shape)
    predictions = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
        pred = model.predict(X_test)
        pred = scaler.inverse_transform(pred)
        predictions[name] = pred
    
    # 3. Evaluate
    y_true = test_values
    results = []
    for name, pred in predictions.items():
        rmse, mae, mape_val, da = evaluate_all_metrics(y_true, pred)
        results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape_val, 'Directional Accuracy (%)': da})
    
    # Print results (optional save to CSV)
    
    # 4. Plot & save figures
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, 'k-', label='Real Price')
    colors = ['blue','red','green','orange','purple']
    for (name, pred), color in zip(predictions.items(), colors):
        plt.plot(pred, '--', color=color, label=name)
    plt.legend()
    plt.savefig('results/comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics plot, etc.
    # ... (similar for other plots)

if __name__ == '__main__':
    main()