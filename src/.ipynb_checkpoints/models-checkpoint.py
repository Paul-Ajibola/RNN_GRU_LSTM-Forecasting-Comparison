"""
Three sequence models -- vanilla RNN, GRU, LSTM -- built with the exact
same hidden size, number of layers, and output head, so any performance
difference we see comes from the recurrent cell itself, not from an
unfair architecture advantage.
"""
import torch
import torch.nn as nn

HIDDEN_SIZE = 32
NUM_LAYERS = 1


class VanillaRNN(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]          # take the final time step's hidden state
        return self.fc(last_step)


class GRUModel(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


MODEL_REGISTRY = {
    "RNN": VanillaRNN,
    "GRU": GRUModel,
    "LSTM": LSTMModel,
}


if __name__ == "__main__":
    # sanity check: same input, all three models should run without error
    dummy = torch.randn(8, 30, 1)  # batch=8, seq_len=30, features=1
    for name, cls in MODEL_REGISTRY.items():
        model = cls()
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output shape {tuple(out.shape)}, params={n_params}")
