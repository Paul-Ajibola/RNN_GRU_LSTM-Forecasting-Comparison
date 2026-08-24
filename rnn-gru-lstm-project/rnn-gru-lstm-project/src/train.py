"""
Trains each of RNN, GRU, and LSTM on the identical data split and
hyperparameter budget, then saves loss curves and predictions for
later comparison and plotting.
"""
import json
import numpy as np
import torch
import torch.nn as nn

from data_prep import prepare_data
from models import MODEL_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WINDOW = 30
SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_model(model, X_train, y_train, X_test, y_test):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    n = X_train_t.shape[0]
    train_losses, test_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}  "
                  f"train_mse={train_losses[-1]:.5f}  test_mse={test_losses[-1]:.5f}")

    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).cpu().numpy().flatten()

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "predictions": final_preds,
    }


def main():
    set_seed()
    X_train, y_train, X_test, y_test, scaler, series = prepare_data(
        "/home/claude/rnn-gru-lstm-project/data/daily-min-temperatures.csv",
        window=WINDOW,
    )

    results = {}
    for name, cls in MODEL_REGISTRY.items():
        print(f"\nTraining {name}...")
        set_seed()  # same init conditions for every architecture
        model = cls()
        result = train_one_model(model, X_train, y_train, X_test, y_test)
        results[name] = result

    # invert scaling back to real degrees C for interpretability
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    for name in results:
        preds_scaled = results[name]["predictions"].reshape(-1, 1)
        results[name]["predictions_real"] = scaler.inverse_transform(preds_scaled).flatten().tolist()
        results[name]["rmse_real_C"] = float(
            np.sqrt(np.mean((results[name]["predictions_real"] - y_test_real) ** 2))
        )

    output = {
        "y_test_real": y_test_real.tolist(),
        "results": {
            name: {
                "train_losses": r["train_losses"],
                "test_losses": r["test_losses"],
                "predictions_real": r["predictions_real"],
                "rmse_real_C": r["rmse_real_C"],
            } for name, r in results.items()
        },
    }

    with open("/home/claude/rnn-gru-lstm-project/outputs/results.json", "w") as f:
        json.dump(output, f)

    print("\n=== Final Test RMSE (°C) ===")
    for name, r in results.items():
        print(f"  {name}: {r['rmse_real_C']:.3f}")


if __name__ == "__main__":
    main()
