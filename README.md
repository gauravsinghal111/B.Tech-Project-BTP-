# Droplet Length Prediction in Flow-Focusing Microfluidic Channels

B.Tech Project | Chemical Engineering | IIT Dharwad

---

## Overview

A machine learning framework to predict the non-dimensional droplet length **L_D** in square and rectangular flow-focusing microfluidic channels operating in the **dripping regime**.

Data was extracted from published literature, covering both laboratory experiments and CFD simulations across a wide range of fluid systems.

---

## Input Features

| Feature | Description | Unit |
|---|---|---|
| W_c | Main channel width | µm |
| W_d | Dispersed phase channel width | µm |
| H | Channel height | µm |
| ρ_c | Continuous phase density | kg/m³ |
| ρ_d | Dispersed phase density | kg/m³ |
| µ_c | Continuous phase viscosity | mPa·s |
| µ_d | Dispersed phase viscosity | mPa·s |
| Q_c | Continuous phase flow rate | µL/h |
| Q_d | Dispersed phase flow rate | µL/h |
| γ | Interfacial tension | mN/m |

**Target:** L_D (non-dimensional droplet length)

---

## Repository Structure

```
├── Combined/
│   ├── ml_model.py          # 8 ML models
│   └── nn_model.py          # PyTorch neural network
│
├── Experimental/
│   ├── ml_model_E.py
│   └── nn_model_E.py
│
└── Simulation/
    ├── ml_model_S.py
    └── nn_model_S.py
```

---

## Models Implemented

**Classical ML** (via scikit-learn):
Random Forest, XGBoost, LightGBM, Gradient Boosting, SVR, MLP, Ridge, Lasso

**Neural Network** (via PyTorch):
4 feedforward architectures with BatchNorm + Dropout — Small [64,32], Medium [128,64,32], Deep [256,128,64,32], Wide [256,128]

All models are evaluated using 5-fold cross-validation + held-out test set (R², RMSE, MAE).

---

## Setup and Usage

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm torch

# Run ML models
python Combined/ml_model.py
python Experimental/ml_model_E.py
python Simulation/ml_model_S.py

# Run neural network
python Combined/nn_model.py
python Experimental/nn_model_E.py
python Simulation/nn_model_S.py
```