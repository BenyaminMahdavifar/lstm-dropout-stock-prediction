import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def directional_accuracy(y_true, y_pred):
    true_diff = np.diff(y_true.flatten())
    pred_diff = np.diff(y_pred.flatten())
    correct = np.sum((true_diff * pred_diff) > 0)
    return correct / len(true_diff) * 100

def evaluate_all_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)
    return rmse, mae, mape_val, da