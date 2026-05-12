"""
ML Model for Droplet Length (LD) Prediction in Microfluidic Channels
BTP Project - Gaurav  |  Simulation Dataset
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

# Always resolve paths relative to this script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
train_raw = pd.read_csv("Training Set_S.csv")
test_raw  = pd.read_csv("Test Set_S.csv")

train_raw = train_raw.dropna(axis=1, how="all")
test_raw  = test_raw.dropna(axis=1, how="all")

train_raw = train_raw.dropna()
test_raw  = test_raw.dropna(subset=["LD"])

print(f"Training samples : {len(train_raw)}")
print(f"Test     samples : {len(test_raw)}")

# ─────────────────────────────────────────────
# 2. DEFINE FEATURES (exclude Qd/Qc)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    c for c in train_raw.columns
    if c not in ("LD", "Qd/Qc")
]
TARGET = "LD"

X_train = train_raw[FEATURE_COLS].values.astype(float)
y_train = train_raw[TARGET].values.astype(float)
X_test  = test_raw[FEATURE_COLS].values.astype(float)
y_test  = test_raw[TARGET].values.astype(float)

FEAT_NAMES = [
    "Wc", "Wd", "H",
    "ρc", "ρd",
    "μc", "μd",
    "Qc", "Qd",
    "γ"
]

print(f"\nFeatures used ({len(FEATURE_COLS)}):")
for fn, fc in zip(FEAT_NAMES, FEATURE_COLS):
    print(f"  {fn:4s}  <-  {fc}")

# ─────────────────────────────────────────────
# 3. SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. MODEL DEFINITIONS
# ─────────────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=1, random_state=42, n_jobs=-1
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbosity=0
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05,
        num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbose=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        random_state=42
    ),
    "SVR": SVR(kernel="rbf", C=100, epsilon=0.05, gamma="scale"),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu", max_iter=2000,
        learning_rate_init=0.001, random_state=42,
        early_stopping=True, validation_fraction=0.1
    ),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=5000),
}

SCALE_MODELS = {"SVR", "MLP", "Ridge", "Lasso"}

# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION + TEST EVALUATION
# ─────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
predictions = {}

print("\n" + "="*70)
print(f"{'Model':<22} {'CV R²':>8} {'CV RMSE':>10} {'Test R²':>8} {'Test RMSE':>10} {'Test MAE':>9}")
print("="*70)

for name, model in models.items():
    Xtr = X_train_sc if name in SCALE_MODELS else X_train
    Xte = X_test_sc  if name in SCALE_MODELS else X_test

    cv_r2   = cross_val_score(model, Xtr, y_train, cv=kf, scoring="r2", n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(model, Xtr, y_train, cv=kf,
                                       scoring="neg_mean_squared_error", n_jobs=-1))

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    results[name] = {
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "cv_rmse_mean": cv_rmse.mean(), "cv_rmse_std": cv_rmse.std(),
        "test_r2": r2, "test_rmse": rmse, "test_mae": mae
    }
    predictions[name] = y_pred

    print(f"{name:<22} {cv_r2.mean():>8.4f} {cv_rmse.mean():>10.4f} "
          f"{r2:>8.4f} {rmse:>10.4f} {mae:>9.4f}")

print("="*70)

# ─────────────────────────────────────────────
# 6. IDENTIFY BEST MODEL
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_r2"])
best_res  = results[best_name]
print(f"\nBest model: {best_name}")
print(f"  Test R²  = {best_res['test_r2']:.4f}")
print(f"  Test RMSE= {best_res['test_rmse']:.4f}")
print(f"  Test MAE = {best_res['test_mae']:.4f}")

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# --- 7a. Model comparison bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = list(results.keys())
r2_vals   = [results[m]["test_r2"]   for m in model_names]
rmse_vals = [results[m]["test_rmse"] for m in model_names]
mae_vals  = [results[m]["test_mae"]  for m in model_names]

colors = ["#2ecc71" if m == best_name else "#3498db" for m in model_names]

for ax, vals, title, ylabel in zip(
    axes,
    [r2_vals, rmse_vals, mae_vals],
    ["Test R²", "Test RMSE", "Test MAE"],
    ["R²", "RMSE", "MAE"]
):
    bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005*max(abs(v) for v in vals),
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Model Comparison on Test Set (Simulation)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("S_01_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: S_01_model_comparison.png")

# --- 7b. Cross-validation scores ---
fig, ax = plt.subplots(figsize=(12, 5))
cv_means = [results[m]["cv_r2_mean"] for m in model_names]
cv_stds  = [results[m]["cv_r2_std"]  for m in model_names]
colors2  = ["#e74c3c" if m == best_name else "#9b59b6" for m in model_names]

ax.bar(model_names, cv_means, yerr=cv_stds, color=colors2,
       capsize=5, edgecolor="white", linewidth=0.6)
ax.set_title("5-Fold Cross-Validation R² (Simulation Training Set)", fontweight="bold")
ax.set_ylabel("CV R²")
ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("S_02_cv_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: S_02_cv_scores.png")

# --- 7c. Actual vs Predicted – all models (grid) ---
ncols = 4
nrows = int(np.ceil(len(models) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    ax = axes[i]
    yp = predictions[name]
    r2 = results[name]["test_r2"]
    ax.scatter(y_test, yp, alpha=0.6, s=25, edgecolors="none",
               color="#e74c3c" if name == best_name else "#3498db")
    lims = [min(y_test.min(), yp.min()) - 0.1, max(y_test.max(), yp.max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"{name}\nR²={r2:.4f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Actual LD", fontsize=8)
    ax.set_ylabel("Predicted LD", fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Actual vs Predicted LD — All Models (Simulation)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("S_03_actual_vs_predicted_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: S_03_actual_vs_predicted_all.png")

# --- 7d. Best model – detailed actual vs predicted + residuals ---
y_pred_best = predictions[best_name]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sc = ax.scatter(y_test, y_pred_best, c=y_test, cmap="viridis",
                alpha=0.75, s=40, edgecolors="none")
lims = [min(y_test.min(), y_pred_best.min()) - 0.2,
        max(y_test.max(), y_pred_best.max()) + 0.2]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual LD", fontsize=11)
ax.set_ylabel("Predicted LD", fontsize=11)
ax.set_title(f"{best_name} — Actual vs Predicted\nR²={best_res['test_r2']:.4f}  "
             f"RMSE={best_res['test_rmse']:.4f}  MAE={best_res['test_mae']:.4f}",
             fontsize=10, fontweight="bold")
plt.colorbar(sc, ax=ax, label="Actual LD")
ax.legend(fontsize=9)

residuals = y_test - y_pred_best
ax2 = axes[1]
ax2.scatter(y_pred_best, residuals, alpha=0.65, s=35,
            c=np.abs(residuals), cmap="RdYlGn_r", edgecolors="none")
ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Predicted LD", fontsize=11)
ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
ax2.set_title("Residual Plot", fontsize=10, fontweight="bold")
ax2.text(0.97, 0.97, f"Bias: {residuals.mean():.4f}\nStd: {residuals.std():.4f}",
         transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
         fontsize=9)

plt.tight_layout()
plt.savefig("S_04_best_model_detail.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: S_04_best_model_detail.png")

# --- 7e. Feature importance ---
fi_model_name = best_name if best_name not in ("SVR", "MLP", "Ridge", "Lasso") else "Random Forest"
fi_model = models[fi_model_name]

if hasattr(fi_model, "feature_importances_"):
    importances = fi_model.feature_importances_
    imp_type = "Built-in Impurity Importance"
else:
    rf_fi = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_fi.fit(X_train, y_train)
    importances = rf_fi.feature_importances_
    fi_model_name = "Random Forest"
    imp_type = "Built-in Impurity Importance"

order = np.argsort(importances)[::-1]
sorted_feats = [FEAT_NAMES[i] for i in order]
sorted_vals  = importances[order]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(sorted_feats, sorted_vals,
              color=sns.color_palette("Blues_r", len(FEAT_NAMES)),
              edgecolor="white", linewidth=0.5)
ax.set_title(f"Feature Importance — {fi_model_name}\n({imp_type})",
             fontweight="bold")
ax.set_ylabel("Importance Score")
ax.set_xlabel("Feature")
for bar, val in zip(bars, sorted_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("S_05_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: S_05_feature_importance.png")

# --- 7f. LD Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
axes[0].set_title("LD Distribution — Simulation Training Set", fontweight="bold")
axes[0].set_xlabel("LD"); axes[0].set_ylabel("Count")

axes[1].hist(y_test, bins=25, color="#e67e22", edgecolor="white", alpha=0.85)
axes[1].set_title("LD Distribution — Simulation Test Set", fontweight="bold")
axes[1].set_xlabel("LD"); axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("S_06_ld_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: S_06_ld_distribution.png")

# --- 7g. Correlation heatmap ---
corr_df = train_raw[FEATURE_COLS + [TARGET]].copy()
corr_df.columns = FEAT_NAMES + ["LD"]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            mask=np.zeros_like(corr_df.corr(), dtype=bool))
ax.set_title("Pearson Correlation Heatmap (Simulation Training Set)", fontweight="bold")
plt.tight_layout()
plt.savefig("S_07_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: S_07_correlation_heatmap.png")

# --- 7h. Parity plot with ±10% bands ---
fig, ax = plt.subplots(figsize=(7, 6))
abs_err = np.abs(y_test - y_pred_best)
sc = ax.scatter(y_test, y_pred_best, c=abs_err, cmap="hot_r",
                s=45, alpha=0.8, edgecolors="none")
lims2 = [0, max(y_test.max(), y_pred_best.max()) + 0.5]
ax.plot(lims2, lims2, "b-", linewidth=1.5, label="Ideal")
ax.plot(lims2, [l * 1.1 for l in lims2], "b--", linewidth=0.8, alpha=0.5, label="+10%")
ax.plot(lims2, [l * 0.9 for l in lims2], "b--", linewidth=0.8, alpha=0.5, label="-10%")
ax.set_xlim(lims2); ax.set_ylim(lims2)
ax.set_xlabel("Actual LD", fontsize=12)
ax.set_ylabel("Predicted LD", fontsize=12)
ax.set_title(f"Parity Plot — {best_name}\n(colour = absolute error)", fontweight="bold")
plt.colorbar(sc, ax=ax, label="Abs. Error")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("S_08_parity_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: S_08_parity_plot.png")

# ─────────────────────────────────────────────
# 8. SUMMARY TABLE TO CSV
# ─────────────────────────────────────────────
summary_rows = []
for name, res in results.items():
    summary_rows.append({
        "Model": name,
        "CV R² (mean)": round(res["cv_r2_mean"], 4),
        "CV R² (std)":  round(res["cv_r2_std"],  4),
        "CV RMSE (mean)": round(res["cv_rmse_mean"], 4),
        "CV RMSE (std)":  round(res["cv_rmse_std"],  4),
        "Test R²":   round(res["test_r2"],   4),
        "Test RMSE": round(res["test_rmse"], 4),
        "Test MAE":  round(res["test_mae"],  4),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Test R²", ascending=False)
summary_df.to_csv("S_model_results_summary.csv", index=False)
print("\nSaved: S_model_results_summary.csv")

# ─────────────────────────────────────────────
# 9. SAVE TEST PREDICTIONS
# ─────────────────────────────────────────────
pred_df = test_raw[FEATURE_COLS + [TARGET]].copy()
for name, yp in predictions.items():
    pred_df[f"Pred_{name}"] = np.round(yp, 4)
pred_df.to_csv("S_test_predictions.csv", index=False)
print("Saved: S_test_predictions.csv")

print("\nDone.")
