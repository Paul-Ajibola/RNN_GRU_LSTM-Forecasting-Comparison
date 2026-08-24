"""
Data preparation for sequence models.

Loads the daily-min-temperature series, scales it, and converts it into
sliding-window (X, y) sequences suitable for RNN/GRU/LSTM training.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df.set_index("Date")["Temp"]


def make_windows(values: np.ndarray, window: int):
    """Turn a 1D array into overlapping (X, y) sequence windows."""
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    return np.array(X), np.array(y)


def prepare_data(csv_path: str, window: int = 30, test_frac: float = 0.15):
    """
    Returns scaled train/test tensors ready for PyTorch, plus the fitted
    scaler (needed later to invert predictions back to real temperatures).

    Uses a chronological split (not random) because shuffling time series
    data leaks future information into training -- a common mistake worth
    explicitly avoiding and mentioning in an interview.
    """
    series = load_series(csv_path)
    values = series.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values).flatten()

    X, y = make_windows(scaled, window)

    split_idx = int(len(X) * (1 - test_frac))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # shape: (samples, seq_len, n_features=1)
    X_train = X_train.reshape(-1, window, 1)
    X_test = X_test.reshape(-1, window, 1)

    return X_train, y_train, X_test, y_test, scaler, series


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler, series = prepare_data(
        "/home/claude/rnn-gru-lstm-project/data/daily-min-temperatures.csv"
    )
    print(f"Series length: {len(series)}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
