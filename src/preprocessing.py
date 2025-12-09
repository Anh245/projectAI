import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, data):
        """Chuẩn hóa dữ liệu train"""
        return self.scaler.fit_transform(data)

    def transform(self, data):
        """Chuẩn hóa dữ liệu test dựa trên scaler đã fit"""
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """Chuyển ngược từ 0-1 về giá thực"""
        return self.scaler.inverse_transform(data)

    def create_dataset(self, dataset, look_back=50):
        """
        Tạo dữ liệu dạng chuỗi thời gian cho LSTM.
        Look_back: số ngày quá khứ dùng để dự đoán ngày tiếp theo.
        """
        X, Y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i - look_back:i, 0])
            Y.append(dataset[i, 0])

        X, Y = np.array(X), np.array(Y)
        # Reshape lại cho đúng input của LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y = np.reshape(Y, (Y.shape[0], 1))
        return X, Y