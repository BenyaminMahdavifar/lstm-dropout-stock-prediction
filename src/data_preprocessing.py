import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_scale_data(train_path='data/train.csv', test_path='data/test.csv', feature_col='Open'):
    """Load train and test CSV, scale the 'Open' column, return scaled training set and scaler."""
    # Load
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_values = train_df[[feature_col]].values
    test_values = test_df[[feature_col]].values
    
    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values)
    
    return train_scaled, test_values, scaler

def create_sequences(data, timesteps=60):
    """Create input sequences and corresponding targets."""
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def prepare_test_data(train_scaled, test_values, scaler, timesteps=60):
    """Prepare test sequences using a concatenated series."""
    # Flatten
    train_flat = train_scaled.flatten()
    test_flat = test_values.flatten()
    
    # Concatenate
    combined = np.concatenate([train_flat, test_flat])
    
    # Take the last part: len(test) + timesteps
    inputs = combined[-(len(test_flat) + timesteps):]
    inputs = inputs.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)
    
    X_test = []
    for i in range(timesteps, len(inputs_scaled)):
        X_test.append(inputs_scaled[i - timesteps:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_test