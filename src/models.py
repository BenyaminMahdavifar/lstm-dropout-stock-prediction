from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam

def build_ascending_model(input_shape):
    model = Sequential([
        Input(input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=True),
        Dropout(0.4),
        LSTM(50),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mean_squared_error')
    return model

def build_descending_model(input_shape):
    model = Sequential([
        Input(input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.5),
        LSTM(50, return_sequences=True),
        Dropout(0.4),
        LSTM(50, return_sequences=True),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mean_squared_error')
    return model

def build_hourglass_model(input_shape):
    model = Sequential([
        Input(input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.5),
        LSTM(50, return_sequences=True),
        Dropout(0.5),
        LSTM(50),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mean_squared_error')
    return model

def build_ascending_bottleneck_model(input_shape):
    model = Sequential([
        Input(input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.4),
        LSTM(50, return_sequences=True),
        Dropout(0.6),
        LSTM(50),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mean_squared_error')
    return model

def build_extra_lstm_model(input_shape):
    model = Sequential([
        Input(input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.1),
        LSTM(50, return_sequences=True),
        Dropout(0.4),
        LSTM(50, return_sequences=True),  # extra LSTM
        LSTM(50, return_sequences=True),
        Dropout(0.6),
        LSTM(50),
        Dropout(0.1),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mean_squared_error')
    return model

def get_all_models(input_shape):
    return {
        'Ascending (0.2->0.5)': build_ascending_model(input_shape),
        'Descending (0.5->0.2)': build_descending_model(input_shape),
        'Hourglass (0.2->0.5->0.2)': build_hourglass_model(input_shape),
        'Ascending Bottleneck (0.4->0.6)': build_ascending_bottleneck_model(input_shape),
        'Expansion LSTM (extra LSTM)': build_extra_lstm_model(input_shape)
    }