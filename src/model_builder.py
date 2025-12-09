from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import load_model
import os


def build_lstm_model(input_shape):
    """Xây dựng kiến trúc model với regularization tốt hơn"""
    model = Sequential()
    # Lớp LSTM 1 - Giảm units và thêm dropout
    model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    
    # Lớp LSTM 2
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Output
    model.add(Dense(1))

    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


def load_trained_model(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        return None