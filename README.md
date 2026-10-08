# Machine Learning Assignment 1: Polynomial Regression
**Student Name:** Ayush Patel  
**Roll Number:** BT2024054  
**Course:** Machine Learning (Assignment 1)  

---

## 📌 Executive Summary

This repository contains the end-to-end implementation of polynomial regression models developed for two personalized prediction tasks assigned to roll number **BT2024054**:

1. **Task 1 (`var1`): Net Power Score** — Predicting power generation turbine performance using 6 operational variables.
2. **Task 2 (`var2`): Thermal Anomaly Score** — Predicting geothermal temperature variations using 3 continuous spatial coordinates.

Through an iterative, data-driven journey across **4 model architectures**, we transitioned from a naive baseline with severe underfitting ($R^2 \approx 0.10 - 0.18$) to an optimal regularized solution achieving exceptional generalization accuracy on 5-fold cross-validation:
- **`var1` Final:** **$R^2 = 0.9689$** | **$\text{MSE} = 0.3218$** (Degree 5 Polynomial + Lasso L1 regularization)
- **`var2` Final:** **$R^2 = 0.9929$** | **$\text{MSE} = 0.2835$** (Degree 9 Polynomial + Ridge L2 regularization)

---

## 📊 Complete Experimental Progression (All 4 Models)

To systematically explore the bias-variance spectrum, four distinct models were developed, trained, cross-validated, and saved:

| Model Stage | Description & Architecture | var1 CV $R^2$ | var1 CV MSE | var2 CV $R^2$ | var2 CV MSE | Status / Key Insight |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Model 1: Baseline** (`model1_baseline.py`) | **PDF Hint Model (OLS)**<br>• var1: Degree 3, $x_1-x_3$ only<br>• var2: Degree 4, $x_1$ only | `0.1838` | `8.9044` | `0.1058` | `36.678` | **Severe Underfitting**: Generic problem hints discarded critical features and under-specified degrees. |
| **Model 2: OLS Sweep** (`model2_ols_sweep.py`) | **Data-Driven Unregularized OLS**<br>• var1: Degree 4, all 6 features ($x_1-x_6$)<br>• var2: Degree 8, all 3 features ($x_1-x_3$) | `0.9230` | `0.8081` | `0.9927` | `0.2896` | **Massive Leap**: Restoring all features and finding the natural degree jumped performance to $>0.92$. |
| **Model 3: Regularized** (`model3_regularized.py`) | **Lasso & Ridge Regularization**<br>• var1: Degree 5 + Lasso ($L_1$, $\alpha \approx 0.0033$)<br>• var2: Degree 9 + Ridge ($L_2$, $\alpha \approx 0.0066$) | `0.9686` | `0.3335` | `0.9929` | `0.2835` | **Controlled Complexity**: Enabled higher degrees while preventing noise fitting via penalty shrinkage. |
| **Model 3+ (Tuned Final)** (`train_predict.py`) | **Fine-Tuned Optimal Regularization**<br>• var1: Degree 5 + Lasso ($\alpha = 0.00348$ via CV zoom)<br>• var2: Degree 9 + Ridge ($\alpha = 0.006649$ via RidgeCV) | **`0.9689`** | **`0.3218`** | **`0.9929`** | **`0.2835`** | **WINNER (Final Submit)**: Peak generalization score with verified minimum mean squared error. |
| **Model 4: Overfit Demo** (`model4_overfit.py`) | **Extreme Overfitting (Unconstrained OLS)**<br>• var1: Degree 8 OLS (3,003 features!)<br>• var2: Degree 15 OLS (816 features!) | `0.6757`<br>*(Train: 0.9998)* | `3.414`<br>*(Train: 0.002)* | `-4.906`<br>*(Train: 0.9999)* | `285.50`<br>*(Train: 0.0001)* | **Catastrophic Overfitting**: Memorized training set ($R^2 \approx 1.0$) but collapsed completely on unseen validation folds. |

---

## 🧠 Thought Process & Technical Justifications (In Plain English)

### 1. Why did the initial PDF hints fail?
The assignment PDF suggested starting with degree 3 using $x_1-x_3$ for `var1`, and degree 4 using only $x_1$ for `var2`. Following this blindly produced terrible accuracy ($R^2 \approx 0.18$ and $0.10$). In a 6-variable power turbine, ignoring half the operational variables leaves out essential thermodynamics. In a 3-dimensional geothermal heat field, ignoring 2 of the 3 spatial coordinates makes accurate temperature prediction impossible. Each student's dataset is independently generated, proving that empirical cross-validation must guide feature selection.

### 2. Why use Lasso ($L_1$) for `var1` (Net Power Score)?
When expanding 6 input variables to Degree 5, we generate **462 polynomial features**. In physical turbines, not all 462 higher-order cross-interaction terms actually exist in the physical system. Lasso applies an absolute-value penalty ($\alpha \sum |\beta_j|$) that forces negligible coefficients **strictly to zero**. It effectively acts as an automated feature selector, stripping away noise and leaving only the genuine physical interactions. Ridge ($L_2$) leaves all 462 weights non-zero and could only achieve $R^2 = 0.952$, whereas Lasso reached **$0.9689$**.

### 3. Why use Ridge ($L_2$) for `var2` (Thermal Anomaly Score)?
Heat diffusion through a continuous subterranean reservoir follows smooth, continuous mathematical laws (governed by the Laplace/heat diffusion equations). There are only 3 spatial inputs ($x_1, x_2, x_3$), and at Degree 9 there are **220 polynomial features**. Because heat varies continuously across all three dimensions, zeroing out terms (like Lasso does) disrupts smooth spatial curvature. Ridge applies a squared penalty ($\alpha \sum \beta_j^2$) that shrinks all weights smoothly, preventing extreme oscillations while keeping all spatial harmonics intact.

### 4. What does Model 4 prove?
Model 4 was deliberately constructed to demonstrate what happens when polynomial regression is pushed too far without regularization. At Degree 15 for `var2`, the model has enough capacity to weave through every single training point ($R^2 = 0.9999$, $\text{MSE} \approx 0$). However, on validation splits, the curve oscillates wildly between points (Runge's phenomenon), causing validation MSE to explode to **285.5** and $R^2$ to crash to **-4.906**. This empirically confirms that Model 3 sits right at the optimal point of the bias-variance tradeoff.

---

## 📁 Repository Structure

```
ML_ASSIGNMENT/
├── BT2024054/
│   ├── BT2024054_train_var1.csv          # Training data: Net Power Score (1001 × 7)
│   ├── BT2024054_test_var1.csv           # Test inputs: Net Power Score (1000 × 6)
│   ├── BT2024054_train_var2.csv          # Training data: Thermal Anomaly (1001 × 4)
│   ├── BT2024054_test_var2.csv           # Test inputs: Thermal Anomaly (1000 × 3)
│   ├── BT2024054_pred_var1.csv           # ★ FINAL SUBMISSION: Predictions for var1 (Model 3+)
│   ├── BT2024054_pred_var2.csv           # ★ FINAL SUBMISSION: Predictions for var2 (Model 3+)
│   ├── BT2024054_pred_var1_m1.csv        # Baseline predictions (Model 1)
│   ├── BT2024054_pred_var2_m1.csv        # Baseline predictions (Model 1)
│   ├── BT2024054_pred_var1_m2.csv        # Intermediate OLS predictions (Model 2)
│   ├── BT2024054_pred_var2_m2.csv        # Intermediate OLS predictions (Model 2)
│   ├── BT2024054_pred_var1_m4.csv        # Overfit predictions (Model 4)
│   └── BT2024054_pred_var2_m4.csv        # Overfit predictions (Model 4)
├── train_predict.py                      # ★ Main production script (Generates final submission)
├── generate_plots.py                     # Generates all 5 publication-quality figures
├── model1_baseline.py                    # Stage 1: Naive PDF-hint baseline implementation
├── model2_ols_sweep.py                   # Stage 2: Data-driven OLS feature & degree sweep
├── model3_regularized.py                 # Stage 3: Regularized Lasso & Ridge model
├── model4_overfit.py                     # Stage 4: High-degree intentional overfit demonstration
├── figures/                              # Generated high-resolution diagnostic plots (300 DPI)
│   ├── model_comparison.png              # Comparison of R² and MSE across all 4 stages
│   ├── bias_variance_tradeoff.png        # Degree sweeps & U-shaped error curves
│   ├── actual_vs_predicted.png           # Ground truth vs predicted scatter with 1:1 line
│   ├── residual_diagnostics.png          # Error distributions & homoscedasticity checks
│   └── sparsity_and_coefficients.png     # Lasso sparsity pruning & Ridge shrinkage spectrum
├── REPORT.md                             # Comprehensive technical report (Markdown format)
└── README.md                             # Repository overview and guide
```

---

## 🚀 How to Run the Code

### 1. Environment Setup
The codebase requires standard scientific Python libraries:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### 2. Generate Final Predictions (Recommended)
To run the winning regularized models and generate the official submission CSVs:
```bash
python3 train_predict.py
```
This script will output:
- `BT2024054/BT2024054_pred_var1.csv`
- `BT2024054/BT2024054_pred_var2.csv`

### 3. Generate Diagnostic Visualizations
To re-generate all high-resolution figures into the `figures/` directory:
```bash
python3 generate_plots.py
```

### 4. Replicate the Experimental Journey (Individual Stages)
Each experimental stage can be executed independently to observe the performance progression:

```bash
# Stage 1: Run naive baseline (PDF hints)
python3 model1_baseline.py

# Stage 2: Run unregularized OLS with all features
python3 model2_ols_sweep.py

# Stage 3: Run standard regularized models (Lasso + Ridge)
python3 model3_regularized.py

# Stage 4: Run extreme overfit demonstration
python3 model4_overfit.py
```

---

## 📈 Visualizations & Diagnostic Figures

The repository includes publication-grade diagnostic plots generated at 300 DPI:

### 1. Model Progression Across Stages
Comparison of Cross-Validation $R^2$ and logarithmic Mean Squared Error across all four model configurations:
![Model Comparison](figures/model_comparison.png)

### 2. Bias-Variance Tradeoff Curves
Validation MSE trajectories showing the U-shaped error curves as polynomial degree increases:
![Bias-Variance Tradeoff](figures/bias_variance_tradeoff.png)

### 3. Actual vs. Predicted Plots
Out-of-fold cross-validated predictions plotted against ground truth labels along the ideal 1:1 reference line:
![Actual vs Predicted](figures/actual_vs_predicted.png)

### 4. Sparsity & Coefficient Shrinkage
Visual proof of Lasso's automated feature selection (pruning 78.8% of terms) and Ridge's smooth weight stabilization:
![Sparsity and Coefficients](figures/sparsity_and_coefficients.png)

### 5. Residual Distribution Diagnostics
Error histograms and residual-vs-fitted plots confirming zero-mean Gaussian distribution and homoscedasticity:
![Residual Diagnostics](figures/residual_diagnostics.png)

---

## 📋 Prediction File Verification

Both generated final prediction files strictly adhere to the assignment guidelines:
- **Header format:** Exactly 1 column named `y` matching `sample_submission.csv`.
- **Row count:** Exactly 1,000 predictions matching the test dataset.
- **Data integrity:** No NaN, null, or infinite values.
- **Value distribution:** Prediction distributions closely match the empirical training distributions (`var1` mean: $\sim 0.76$, `var2` mean: $\sim 2.40$).

---

## 🧑‍💻 Author
- **Ayush Patel** (Roll No: **BT2024054**)  
- Repository: [https://github.com/Ayush-patel9/ML_ASSIGNMENT](https://github.com/Ayush-patel9/ML_ASSIGNMENT)
