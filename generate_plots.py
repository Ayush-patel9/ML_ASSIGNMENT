"""
Generate Publication-Quality Visualizations for ML Assignment 1 (Polynomial Regression)
Roll Number: BT2024054 | Student: Ayush Patel
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, RidgeCV
from sklearn.model_selection import cross_val_predict, KFold

os.makedirs("figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Load data
tr1 = pd.read_csv("BT2024054/BT2024054_train_var1.csv")
X1 = tr1[["x1", "x2", "x3", "x4", "x5", "x6"]].values
y1 = tr1["y"].values

tr2 = pd.read_csv("BT2024054/BT2024054_train_var2.csv")
X2 = tr2[["x1", "x2", "x3"]].values
y2 = tr2["y"].values

print("Generating Plot 1: Model Comparison Across All 4 Stages...")
# ==============================================================================
# Plot 1: Model Comparison (R² and MSE)
# ==============================================================================
models = ["M1: Baseline\n(PDF Hints)", "M2: OLS Sweep\n(All Features)", "M3+: Tuned Reg\n(WINNER)", "M4: Overfit\n(Excessive Deg)"]
var1_r2 = [0.1838, 0.9230, 0.9689, 0.6757]
var1_mse = [8.9044, 0.8081, 0.3218, 3.4140]

var2_r2 = [0.1058, 0.9927, 0.9929, -4.906]  # Clamped visually for overfit
var2_mse = [36.678, 0.2896, 0.2835, 285.50]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(len(models))
width = 0.35

# Subplot 1: CV R²
rects1 = ax1.bar(x - width/2, var1_r2, width, label="var1 (Net Power)", color="#2b5c8f", edgecolor="black", alpha=0.85)
# Clamp negative value for clean visual rendering with note
var2_r2_plot = [v if v > 0 else -0.5 for v in var2_r2]
rects2 = ax1.bar(x + width/2, var2_r2_plot, width, label="var2 (Thermal Anomaly)", color="#d95f02", edgecolor="black", alpha=0.85)

ax1.set_ylabel("Cross-Validation $R^2$ Score")
ax1.set_title("Cross-Validation $R^2$ Progression Across Model Stages")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(-0.6, 1.08)
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax1.axhline(1.0, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
ax1.legend(loc="upper left")

# Annotate values
for rect, val in zip(rects1, var1_r2):
    ax1.annotate(f"{val:.3f}", (rect.get_x() + rect.get_width() / 2, max(val, 0)),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
for rect, val in zip(rects2, var2_r2):
    txt = f"{val:.3f}" if val > 0 else f"{val:.2f} (Collapses)"
    y_pos = max(val, -0.45)
    ax1.annotate(txt, (rect.get_x() + rect.get_width() / 2, y_pos),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom" if val > 0 else "top",
                 fontsize=9, fontweight="bold", color="darkred" if val < 0 else "black")

# Subplot 2: CV MSE (Log Scale)
rects3 = ax2.bar(x - width/2, var1_mse, width, label="var1 MSE", color="#2b5c8f", edgecolor="black", alpha=0.85)
rects4 = ax2.bar(x + width/2, var2_mse, width, label="var2 MSE", color="#d95f02", edgecolor="black", alpha=0.85)
ax2.set_ylabel("Cross-Validation Mean Squared Error (Log Scale)")
ax2.set_yscale("log")
ax2.set_title("Cross-Validation MSE Across Stages (Lower is Better)")
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend(loc="upper left")

for rect, val in zip(rects3, var1_mse):
    ax2.annotate(f"{val:.2f}", (rect.get_x() + rect.get_width() / 2, val),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
for rect, val in zip(rects4, var2_mse):
    ax2.annotate(f"{val:.2f}", (rect.get_x() + rect.get_width() / 2, val),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/model_comparison.png")
plt.close()

print("Generating Plot 2: Bias-Variance Degree Sweeps...")
# ==============================================================================
# Plot 2: Bias-Variance Tradeoff Curves
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# var1 OLS sweep vs Lasso
v1_degs = [1, 2, 3, 4, 5, 6]
v1_mse_ols = [9.7633, 3.5145, 1.1650, 0.8081, 1.6288, 97.3203]
ax1.plot(v1_degs, v1_mse_ols, marker="o", color="#d95f02", linewidth=2, label="OLS (Unregularized)")
ax1.scatter([5], [0.3218], color="#2ca02c", s=140, zorder=5, label="Lasso Deg 5 (WINNER: 0.3218)")
ax1.annotate("Lasso Optimum\n(MSE = 0.32)", (5, 0.3218), xytext=(4.3, 15),
             arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6),
             fontweight="bold", color="darkgreen")
ax1.annotate("OLS Noise Explosion\n(MSE = 97.32)", (6, 97.3203), xytext=(4.6, 60),
             arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
             fontweight="bold", color="darkred")
ax1.set_xlabel("Polynomial Degree")
ax1.set_ylabel("Validation MSE (Log Scale)")
ax1.set_yscale("log")
ax1.set_title("var1 (Net Power): Bias-Variance Curve vs Degree")
ax1.set_xticks(v1_degs)
ax1.legend(loc="upper left")

# var2 OLS sweep vs Ridge
v2_degs = [1, 2, 3, 4, 6, 7, 8, 9, 10]
v2_mse_ols = [32.1021, 19.7887, 11.1305, 3.4457, 0.6158, 0.3794, 0.2896, 0.3372, 0.3871]
ax2.plot(v2_degs, v2_mse_ols, marker="s", color="#2b5c8f", linewidth=2, label="OLS (Unregularized)")
ax2.scatter([9], [0.2835], color="#2ca02c", s=140, zorder=5, label="Ridge Deg 9 (WINNER: 0.2835)")
ax2.annotate("Ridge Optimum\n(MSE = 0.283)", (9, 0.2835), xytext=(7.2, 1.2),
             arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6),
             fontweight="bold", color="darkgreen")
ax2.set_xlabel("Polynomial Degree")
ax2.set_ylabel("Validation MSE (Log Scale)")
ax2.set_yscale("log")
ax2.set_title("var2 (Thermal Anomaly): Bias-Variance Curve vs Degree")
ax2.set_xticks(v2_degs)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("figures/bias_variance_tradeoff.png")
plt.close()

print("Generating Plot 3: Actual vs Predicted...")
# ==============================================================================
# Plot 3: Actual vs Predicted Scatter Plots
# ==============================================================================
# Generate out-of-fold predictions
pf1 = PolynomialFeatures(degree=5)
Xp1 = pf1.fit_transform(X1)
m1_lasso = Lasso(alpha=0.00348, max_iter=30000)
pred_cv1 = cross_val_predict(m1_lasso, Xp1, y1, cv=kf)

pf2 = PolynomialFeatures(degree=9)
Xp2 = pf2.fit_transform(X2)
m2_ridge = RidgeCV(alphas=np.logspace(-4, 4, 80), cv=5)
pred_cv2 = cross_val_predict(m2_ridge, Xp2, y2, cv=kf)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# var1
res1 = np.abs(y1 - pred_cv1)
scatter1 = ax1.scatter(y1, pred_cv1, c=res1, cmap="viridis", alpha=0.6, s=24, edgecolors="none")
min1, max1 = min(y1.min(), pred_cv1.min()), max(y1.max(), pred_cv1.max())
ax1.plot([min1, max1], [min1, max1], "r--", linewidth=1.8, label=r"Ideal 1:1 Reference Line ($y = \hat{y}$)")
ax1.set_title("var1 (Net Power): Actual vs Predicted\n(Degree 5 Lasso | CV $R^2 = 0.9689$, MSE = $0.3218$)")
ax1.set_xlabel("Ground Truth ($y$)")
ax1.set_ylabel(r"Cross-Validated Prediction ($\hat{y}$)")
ax1.legend(loc="upper left")
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label(r"Absolute Error ($|y - \hat{y}|$)")

# var2
res2 = np.abs(y2 - pred_cv2)
scatter2 = ax2.scatter(y2, pred_cv2, c=res2, cmap="plasma", alpha=0.6, s=24, edgecolors="none")
min2, max2 = min(y2.min(), pred_cv2.min()), max(y2.max(), pred_cv2.max())
ax2.plot([min2, max2], [min2, max2], "r--", linewidth=1.8, label=r"Ideal 1:1 Reference Line ($y = \hat{y}$)")
ax2.set_title("var2 (Thermal Anomaly): Actual vs Predicted\n(Degree 9 Ridge | CV $R^2 = 0.9929$, MSE = $0.2835$)")
ax2.set_xlabel("Ground Truth ($y$)")
ax2.set_ylabel(r"Cross-Validated Prediction ($\hat{y}$)")
ax2.legend(loc="upper left")
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label(r"Absolute Error ($|y - \hat{y}|$)")

plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png")
plt.close()

print("Generating Plot 4: Residual Diagnostics...")
# ==============================================================================
# Plot 4: Residual Distributions
# ==============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

err1 = y1 - pred_cv1
err2 = y2 - pred_cv2

# Residual histogram var1
ax1.hist(err1, bins=35, color="#2b5c8f", edgecolor="black", alpha=0.75, density=True)
mu1, std1 = np.mean(err1), np.std(err1)
norm_x1 = np.linspace(err1.min(), err1.max(), 100)
ax1.plot(norm_x1, 1/(std1 * np.sqrt(2*np.pi)) * np.exp(-(norm_x1 - mu1)**2 / (2 * std1**2)), "r-", linewidth=2, label="Normal Fit")
ax1.set_title(rf"var1 Residual Distribution" + "\n" + rf"($\mu = {mu1:.2e}, \sigma = {std1:.3f}$)")
ax1.set_xlabel(r"Residual ($y - \hat{y}$)")
ax1.set_ylabel("Density")
ax1.legend()

# Residual histogram var2
ax2.hist(err2, bins=35, color="#d95f02", edgecolor="black", alpha=0.75, density=True)
mu2, std2 = np.mean(err2), np.std(err2)
norm_x2 = np.linspace(err2.min(), err2.max(), 100)
ax2.plot(norm_x2, 1/(std2 * np.sqrt(2*np.pi)) * np.exp(-(norm_x2 - mu2)**2 / (2 * std2**2)), "r-", linewidth=2, label="Normal Fit")
ax2.set_title(rf"var2 Residual Distribution" + "\n" + rf"($\mu = {mu2:.2e}, \sigma = {std2:.3f}$)")
ax2.set_xlabel(r"Residual ($y - \hat{y}$)")
ax2.set_ylabel("Density")
ax2.legend()

# Residual vs Fitted var1
ax3.scatter(pred_cv1, err1, alpha=0.5, color="#2b5c8f", s=18)
ax3.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax3.set_title(r"var1 Residuals vs Predicted Values (Homoscedasticity)")
ax3.set_xlabel(r"Predicted ($\hat{y}$)")
ax3.set_ylabel(r"Residual ($y - \hat{y}$)")

# Residual vs Fitted var2
ax4.scatter(pred_cv2, err2, alpha=0.5, color="#d95f02", s=18)
ax4.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax4.set_title(r"var2 Residuals vs Predicted Values (Homoscedasticity)")
ax4.set_xlabel(r"Predicted ($\hat{y}$)")
ax4.set_ylabel(r"Residual ($y - \hat{y}$)")

plt.tight_layout()
plt.savefig("figures/residual_diagnostics.png")
plt.close()

print("Generating Plot 5: Sparsity & Weight Regularization...")
# ==============================================================================
# Plot 5: Sparsity & Regularization Effect
# ==============================================================================
# Fit full models to examine weights
m1_lasso.fit(Xp1, y1)
m2_ridge.fit(Xp2, y2)

# Unconstrained overfit models
pf_overfit = PolynomialFeatures(degree=15)
Xp2_overfit = pf_overfit.fit_transform(X2)
m_overfit = LinearRegression().fit(Xp2_overfit, y2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5))

# Top: var1 Lasso coefficients (showing sparsity)
coefs1 = np.abs(m1_lasso.coef_)
active_idx = np.where(coefs1 > 0)[0]
zero_idx = np.where(coefs1 == 0)[0]

ax1.stem(range(len(coefs1)), coefs1, linefmt="C0-", markerfmt="C0o", basefmt="k-")
ax1.set_title(f"var1 Lasso Sparsity: 364 Terms Zeroed Out (78.8%) vs 98 Active Terms (21.2%)\n(Total: 462 Polynomial Features at Degree 5)")
ax1.set_xlabel(r"Polynomial Monomial Index ($0 \dots 461$)")
ax1.set_ylabel(r"Absolute Coefficient ($|w_j|$)")
ax1.annotate("Dense region pruned by L1", xy=(250, 0.05), xytext=(280, 0.4),
             arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=5),
             fontweight="bold")

# Bottom: var2 Ridge vs Overfit OLS coefficients
coefs2_ridge = np.sort(np.abs(m2_ridge.coef_))[::-1]
coefs2_overfit = np.sort(np.abs(m_overfit.coef_))[::-1][:len(coefs2_ridge)]

ax2.plot(range(len(coefs2_ridge)), coefs2_ridge, color="green", linewidth=2, label=f"Model 3+ Ridge Deg 9 (Max: {np.max(coefs2_ridge):.2f}, Norm: 38.9)")
ax2.plot(range(len(coefs2_overfit)), coefs2_overfit, color="red", linestyle="--", linewidth=1.5, label=f"Model 4 OLS Deg 15 (Max: {np.max(coefs2_overfit):.1f}, Norm: 27,038.4)")
ax2.set_yscale("log")
ax2.set_title("var2 Coefficient Shrinkage: Controlled Ridge ($L_2$) vs Astronomical Overfit Explosion")
ax2.set_xlabel("Sorted Feature Rank")
ax2.set_ylabel("Coefficient Magnitude (Log Scale)")
ax2.legend()

plt.tight_layout()
plt.savefig("figures/sparsity_and_coefficients.png")
plt.close()

print("All 5 plots successfully generated and saved to ./figures/")
