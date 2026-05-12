"""
Neural Network for Droplet Length (LD) Prediction in Microfluidic Channels
BTP Project - Gaurav  |  Experimental Dataset
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Dataset-specific settings ─────────────────
TRAIN_FILE    = "Training Set_E.csv"
TEST_FILE     = "Test Set_E.csv"
PREFIX        = "E_"             # output filename prefix
DATASET_LABEL = "Experimental"
# ──────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
train_raw = pd.read_csv(TRAIN_FILE)
test_raw  = pd.read_csv(TEST_FILE)

train_raw = train_raw.dropna(axis=1, how="all")
test_raw  = test_raw.dropna(axis=1, how="all")
train_raw = train_raw.dropna()
test_raw  = test_raw.dropna(subset=["LD"])

print(f"Training samples : {len(train_raw)}")
print(f"Test     samples : {len(test_raw)}")

FEATURE_COLS = [c for c in train_raw.columns if c not in ("LD", "Qd/Qc")]
TARGET       = "LD"
N_FEATURES   = len(FEATURE_COLS)

X_train_np = train_raw[FEATURE_COLS].values.astype(np.float32)
y_train_np = train_raw[TARGET].values.astype(np.float32)
X_test_np  = test_raw[FEATURE_COLS].values.astype(np.float32)
y_test_np  = test_raw[TARGET].values.astype(np.float32)

# ─────────────────────────────────────────────
# 2. NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────
class MLP(nn.Module):
    """Feedforward network with BatchNorm + Dropout after each hidden layer."""
    def __init__(self, input_dim, hidden_layers, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


ARCHITECTURES = {
    "Small [64,32]":          {"hidden": [64, 32],        "dropout": 0.1},
    "Medium [128,64,32]":     {"hidden": [128, 64, 32],   "dropout": 0.2},
    "Deep [256,128,64,32]":   {"hidden": [256,128,64,32], "dropout": 0.2},
    "Wide [256,128]":         {"hidden": [256, 128],      "dropout": 0.2},
}

# ─────────────────────────────────────────────
# 3. TRAINING HELPERS
# ─────────────────────────────────────────────
def train_model(model, X_tr, y_tr, X_val, y_val,
                lr=1e-3, max_epochs=600, patience=50, batch_size=32):
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-5)
    criterion = nn.MSELoss()

    Xtr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(DEVICE)
    ytr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(DEVICE)
    Xval_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    yval_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_w   = None
    no_imp   = 0
    t_hist, v_hist = [], []

    for _ in range(max_epochs):
        model.train()
        ep_loss = sum(
            (lambda loss: (loss.backward(), opt.step(), opt.zero_grad(), loss.item() * len(xb))[-1])(
                criterion(model(xb), yb)
            )
            for xb, yb in loader
        ) / len(Xtr_t)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xval_t), yval_t).item()

        scheduler.step(val_loss)
        t_hist.append(ep_loss)
        v_hist.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp   = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_w:
        model.load_state_dict(best_w)
    return t_hist, v_hist


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()


# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION + TEST EVALUATION
# ─────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results       = {}
predictions   = {}
loss_histories = {}

print("\n" + "="*76)
print(f"{'Architecture':<26} {'CV R²':>8} {'CV RMSE':>10} {'Test R²':>8} {'Test RMSE':>10} {'Test MAE':>9}")
print("="*76)

for arch_name, cfg in ARCHITECTURES.items():
    cv_r2, cv_rmse = [], []

    # 5-fold CV
    for tr_idx, val_idx in kf.split(X_train_np):
        sc = StandardScaler()
        Xtr  = sc.fit_transform(X_train_np[tr_idx])
        Xval = sc.transform(X_train_np[val_idx])
        ytr, yval = y_train_np[tr_idx], y_train_np[val_idx]

        m = MLP(N_FEATURES, cfg["hidden"], cfg["dropout"]).to(DEVICE)
        train_model(m, Xtr, ytr, Xval, yval)
        yp = predict(m, Xval)

        cv_r2.append(r2_score(yval, yp))
        cv_rmse.append(np.sqrt(mean_squared_error(yval, yp)))

    # Full training set → test set
    sc_full = StandardScaler()
    X_tr_sc = sc_full.fit_transform(X_train_np)
    X_te_sc = sc_full.transform(X_test_np)

    val_n = max(1, int(0.1 * len(X_tr_sc)))
    X_tr_fit, X_val_fit = X_tr_sc[:-val_n], X_tr_sc[-val_n:]
    y_tr_fit, y_val_fit = y_train_np[:-val_n], y_train_np[-val_n:]

    m_full = MLP(N_FEATURES, cfg["hidden"], cfg["dropout"]).to(DEVICE)
    t_hist, v_hist = train_model(m_full, X_tr_fit, y_tr_fit, X_val_fit, y_val_fit)

    y_pred = predict(m_full, X_te_sc)
    r2   = r2_score(y_test_np, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
    mae  = mean_absolute_error(y_test_np, y_pred)

    results[arch_name]        = {
        "cv_r2_mean": np.mean(cv_r2),   "cv_r2_std": np.std(cv_r2),
        "cv_rmse_mean": np.mean(cv_rmse), "cv_rmse_std": np.std(cv_rmse),
        "test_r2": r2, "test_rmse": rmse, "test_mae": mae,
    }
    predictions[arch_name]    = y_pred
    loss_histories[arch_name] = (t_hist, v_hist)

    print(f"{arch_name:<26} {np.mean(cv_r2):>8.4f} {np.mean(cv_rmse):>10.4f} "
          f"{r2:>8.4f} {rmse:>10.4f} {mae:>9.4f}")

print("="*76)

best_name = max(results, key=lambda k: results[k]["test_r2"])
best_res  = results[best_name]
print(f"\nBest architecture : {best_name}")
print(f"  Test R²   = {best_res['test_r2']:.4f}")
print(f"  Test RMSE = {best_res['test_rmse']:.4f}")
print(f"  Test MAE  = {best_res['test_mae']:.4f}")

# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
arch_names  = list(results.keys())
short_names = [a.split("[")[0].strip() for a in arch_names]

# --- Plot 1: Architecture comparison ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
r2_vals   = [results[m]["test_r2"]   for m in arch_names]
rmse_vals = [results[m]["test_rmse"] for m in arch_names]
mae_vals  = [results[m]["test_mae"]  for m in arch_names]
colors    = ["#2ecc71" if m == best_name else "#e74c3c" for m in arch_names]

for ax, vals, title, ylabel in zip(
    axes, [r2_vals, rmse_vals, mae_vals],
    ["Test R²", "Test RMSE", "Test MAE"], ["R²", "RMSE", "MAE"]
):
    bars = ax.bar(short_names, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005 * max(abs(v) for v in vals),
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.suptitle(f"Neural Network Architecture Comparison ({DATASET_LABEL})",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{PREFIX}NN_01_arch_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {PREFIX}NN_01_arch_comparison.png")

# --- Plot 2: Training & validation loss curves ---
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()
for i, (name, (t_l, v_l)) in enumerate(loss_histories.items()):
    ax = axes[i]
    ax.plot(t_l, label="Train", color="#3498db", linewidth=1.2)
    ax.plot(v_l, label="Val",   color="#e74c3c", linewidth=1.2, alpha=0.85)
    ax.set_title(short_names[i], fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
plt.suptitle(f"Training & Validation Loss Curves ({DATASET_LABEL})",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{PREFIX}NN_02_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PREFIX}NN_02_loss_curves.png")

# --- Plot 3: Actual vs Predicted – all architectures ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, name in enumerate(arch_names):
    ax  = axes[i]
    yp  = predictions[name]
    r2  = results[name]["test_r2"]
    ax.scatter(y_test_np, yp, alpha=0.65, s=30, edgecolors="none",
               color="#2ecc71" if name == best_name else "#3498db")
    lims = [min(y_test_np.min(), yp.min()) - 0.1,
            max(y_test_np.max(), yp.max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"{short_names[i]}\nR²={r2:.4f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Actual LD", fontsize=8)
    ax.set_ylabel("Predicted LD", fontsize=8)
plt.suptitle(f"Actual vs Predicted — All Architectures ({DATASET_LABEL})",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{PREFIX}NN_03_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PREFIX}NN_03_actual_vs_predicted.png")

# --- Plot 4: Best arch – scatter + residuals ---
y_pred_best = predictions[best_name]
fig, axes   = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sc = ax.scatter(y_test_np, y_pred_best, c=y_test_np, cmap="viridis",
                alpha=0.75, s=40, edgecolors="none")
lims = [min(y_test_np.min(), y_pred_best.min()) - 0.2,
        max(y_test_np.max(), y_pred_best.max()) + 0.2]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual LD", fontsize=11); ax.set_ylabel("Predicted LD", fontsize=11)
ax.set_title(f"Best NN ({best_name}) — Actual vs Predicted\n"
             f"R²={best_res['test_r2']:.4f}  RMSE={best_res['test_rmse']:.4f}  "
             f"MAE={best_res['test_mae']:.4f}", fontsize=9, fontweight="bold")
plt.colorbar(sc, ax=ax, label="Actual LD")
ax.legend(fontsize=9)

residuals = y_test_np - y_pred_best
ax2 = axes[1]
ax2.scatter(y_pred_best, residuals, alpha=0.65, s=35,
            c=np.abs(residuals), cmap="RdYlGn_r", edgecolors="none")
ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Predicted LD", fontsize=11)
ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
ax2.set_title("Residual Plot", fontsize=10, fontweight="bold")
ax2.text(0.97, 0.97,
         f"Bias: {residuals.mean():.4f}\nStd:  {residuals.std():.4f}",
         transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
         fontsize=9)
plt.tight_layout()
plt.savefig(f"{PREFIX}NN_04_best_arch_detail.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PREFIX}NN_04_best_arch_detail.png")

# --- Plot 5: Parity plot ---
fig, ax = plt.subplots(figsize=(7, 6))
abs_err = np.abs(y_test_np - y_pred_best)
sc = ax.scatter(y_test_np, y_pred_best, c=abs_err, cmap="hot_r",
                s=45, alpha=0.8, edgecolors="none")
lims2 = [0, max(y_test_np.max(), y_pred_best.max()) + 0.5]
ax.plot(lims2, lims2, "b-",  linewidth=1.5, label="Ideal")
ax.plot(lims2, [l * 1.1 for l in lims2], "b--", linewidth=0.8, alpha=0.5, label="+10%")
ax.plot(lims2, [l * 0.9 for l in lims2], "b--", linewidth=0.8, alpha=0.5, label="-10%")
ax.set_xlim(lims2); ax.set_ylim(lims2)
ax.set_xlabel("Actual LD", fontsize=12); ax.set_ylabel("Predicted LD", fontsize=12)
ax.set_title(f"Parity Plot — Best NN ({DATASET_LABEL})\n(colour = absolute error)",
             fontweight="bold")
plt.colorbar(sc, ax=ax, label="Abs. Error")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PREFIX}NN_05_parity_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PREFIX}NN_05_parity_plot.png")

# ─────────────────────────────────────────────
# 6. SUMMARY CSV
# ─────────────────────────────────────────────
pd.DataFrame([
    {
        "Architecture":   name,
        "CV R² (mean)":   round(res["cv_r2_mean"],   4),
        "CV R² (std)":    round(res["cv_r2_std"],    4),
        "CV RMSE (mean)": round(res["cv_rmse_mean"], 4),
        "Test R²":        round(res["test_r2"],      4),
        "Test RMSE":      round(res["test_rmse"],    4),
        "Test MAE":       round(res["test_mae"],     4),
    }
    for name, res in results.items()
]).sort_values("Test R²", ascending=False)\
  .to_csv(f"{PREFIX}NN_results_summary.csv", index=False)
print(f"\nSaved: {PREFIX}NN_results_summary.csv")
print("\nDone.")
