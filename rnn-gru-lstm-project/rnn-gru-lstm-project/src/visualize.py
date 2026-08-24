import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/home/rnn-gru-lstm-project/outputs/results.json") as f:
    data = json.load(f)



y_test_real = data["y_test_real"]
results = data["results"]
colors = {"RNN": "#e07a5f", "GRU": "#3d5a80", "LSTM": "#81b29a"}


# --- Plot 1: Predictions vs Actual (last 100 test points, for readability) ---
fig, ax = plt.subplots(figsize=(11, 5))
n_show = 100
ax.plot(y_test_real[-n_show:], label="Actual", color="black", linewidth=2, alpha=0.8)
for name, r in results.items():
    ax.plot(r["predictions_real"][-n_show:], label=f"{name} (RMSE={r['rmse_real_C']:.2f}°C)",
            color=colors[name], linewidth=1.5, alpha=0.85)
ax.set_title("Predicted vs Actual Daily Min Temperature (last 100 test days)")
ax.set_xlabel("Day")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/rnn-gru-lstm-project/outputs/predictions_comparison.png", dpi=150)
plt.close(fig)



# --- Plot 2: Training curves (test loss per epoch) ---
fig, ax = plt.subplots(figsize=(11, 5))
for name, r in results.items():
    ax.plot(r["test_losses"], label=name, color=colors[name], linewidth=1.8)
ax.set_title("Test Loss (MSE, scaled) per Epoch — Convergence Comparison")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (scaled 0-1)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/rnn-gru-lstm-project/outputs/training_curves.png", dpi=150)
plt.close(fig)



# --- Plot 3: Final RMSE bar chart ---
fig, ax = plt.subplots(figsize=(6, 5))
names = list(results.keys())
rmses = [results[n]["rmse_real_C"] for n in names]
bars = ax.bar(names, rmses, color=[colors[n] for n in names])
ax.set_title("Final Test RMSE by Architecture")
ax.set_ylabel("RMSE (°C)")
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center")
fig.tight_layout()
fig.savefig("/home/claude/rnn-gru-lstm-project/outputs/rmse_comparison.png", dpi=150)
plt.close(fig)

print("Saved 3 plots to outputs/")
