# Machine Learning Assignment 1: Polynomial Regression
**Student:** Ayush Patel  
**Roll Number:** BT2024054  
**Assigned Problems:** `var1` (Net Power Score) & `var2` (Thermal Anomaly Score)  
**GitHub Repository:** [https://github.com/Ayush-patel9/ML_ASSIGNMENT](https://github.com/Ayush-patel9/ML_ASSIGNMENT)  

---

## 1. Introduction & Overview

In this assignment, I worked on developing polynomial regression models for two engineering prediction problems assigned to my roll number (**BT2024054**):

1. **Problem 1 (`var1`): Steam Turbine Net Power Score** — Predicting the electrical output of a multi-stage steam turbine from 6 operational control variables (valve positions, flow rates, blade pitch, and pressures).
2. **Problem 2 (`var2`): Subterranean Thermal Anomaly Score** — Mapping underground geothermal temperature anomalies from 3 spatial coordinates (East-West offset, North-South offset, and depth).

The assignment required using **only polynomial regression** to model these systems. Non-polynomial methods (like tree ensembles or neural networks) were not permitted.

When I started by testing the initial suggestions mentioned in the problem PDF (degree 3 on $x_1-x_3$ for `var1`, and degree 4 on $x_1$ for `var2`), the validation performance was very poor ($R^2 \approx 0.18$ and $0.10$). To find the right models, I stepped through four stages of experiments:

- **Stage 1 (Baseline):** Tested the PDF hints directly with Ordinary Least Squares (OLS). Both models underfitted severely.
- **Stage 2 (OLS Sweep):** Ran an exhaustive 5-fold cross-validation sweep over degrees and feature subsets. Bringing in all features boosted $R^2$ to 0.9230 on `var1` and 0.9927 on `var2`, but unregularized OLS hit a wall at higher degrees.
- **Stage 3 & 3+ (Regularized Models — Final Choice):** Introduced Lasso ($L_1$) and Ridge ($L_2$). Matching the regularization type to the underlying physics (Lasso for the 6-variable turbine, Ridge for the 3D thermal field) and fine-tuning the penalty $\alpha$ gave the best generalization:
  - **`var1` Final:** **$\text{CV } R^2 = \mathbf{0.9689}$**, **$\text{CV MSE} = \mathbf{0.3218}$** (Degree 5 + Lasso, $\alpha = 0.00348$)
  - **`var2` Final:** **$\text{CV } R^2 = \mathbf{0.9929}$**, **$\text{CV MSE} = \mathbf{0.2835}$** (Degree 9 + Ridge, $\alpha = 0.00665$)
- **Stage 4 (Overfitting Demo):** Pushed polynomial degrees to extremes without regularization (degree 8 on `var1`, degree 15 on `var2`) to observe where the models break. Training error dropped near zero, but validation error exploded ($\text{CV } R^2 = -4.906$ on `var2`), proving that Model 3 hits the right balance.

---

## 2. Exploring the Datasets

Before training any models, I checked the dataset properties and verified distributions.

### 2.1 Problem 1: `var1` — Turbine Net Power Score
- **Physics:** Net power output depends on thermodynamic variables across the turbine cycle:
  - $x_1$: High-pressure steam valve adjustment
  - $x_2$: Condenser coolant flow rate adjustment
  - $x_3$: Re-injection pump hydraulic pressure
  - $x_4$: Turbine blade pitch angle
  - $x_5$: Non-condensable gas exhaust valve rate
  - $x_6$: Steam inlet pressure adjustment
- **Data shape:** 1,000 training rows $\times$ 7 columns (6 features + target $y$), and 1,000 test rows $\times$ 6 features.
- **Ranges:** All 6 features are pre-normalized between $[-1.0, 1.0]$.
- **Target ($y$):** Training mean $\mu = 0.76$, standard deviation $\sigma = 3.31$, min $-10.43$, max $11.48$.
- **Intuition:** In a physical turbine, variables interact (for example, steam pressure $\times$ valve opening), but arbitrary 5-variable multiplications rarely represent real physical effects. This suggested that a higher-degree polynomial model would need to be **sparse**.

### 2.2 Problem 2: `var2` — Geothermal Thermal Anomaly Score
- **Physics:** Sensor readings at 3D spatial coordinates $(x_1, x_2, x_3)$ relative to basecamp:
  - $x_1$: East-West offset (meters)
  - $x_2$: North-South offset (meters)
  - $x_3$: Vertical depth offset (meters)
- **Data shape:** 1,000 training rows $\times$ 4 columns (3 features + target $y$), and 1,000 test rows $\times$ 3 features.
- **Ranges:** Spatial offsets are normalized between $[-1.0, 1.0]$.
- **Target ($y$):** Training mean $\mu = 2.40$, standard deviation $\sigma = 6.41$, min $-29.69$, max $39.25$.
- **Intuition:** Temperature variation through subterranean rock follows continuous heat conduction (governed by Laplace's heat equation $\nabla^2 T = 0$ in steady state). Heat gradients change smoothly across all three dimensions, so the polynomial needs to be **dense and smooth**, rather than sparse.

---

## 3. Polynomial Expansion & Regularization

Polynomial regression transforms $d$ input features into higher-order combinations up to degree $D$:

$$\hat{y} = \sum_{j=1}^{P} w_j \phi_j(\mathbf{x}) + b$$

The total number of terms $P$ grows quickly with degree $D$ according to:

$$P = \binom{d + D}{D} = \frac{(d + D)!}{d! \, D!}$$

For `var1` (6 features):
- Degree 1: 7 terms
- Degree 2: 28 terms
- Degree 3: 84 terms
- Degree 4: 210 terms
- Degree 5: **462 terms**
- Degree 8: **3,003 terms** (exceeds the 1,000 training rows!)

For `var2` (3 features):
- Degree 4: 35 terms
- Degree 8: 165 terms
- Degree 9: **220 terms**
- Degree 15: **816 terms**

When $P$ gets large relative to $N$, Ordinary Least Squares (OLS) easily fits noise in the training set. Regularization adds a penalty on the weights $\mathbf{w}$ to prevent this:

- **Lasso ($L_1$ penalty):** $\text{Loss} = \text{MSE} + \alpha \sum |w_j|$. The $L_1$ diamond penalty forces less important coefficients to become exactly 0. It acts as an automatic feature selector, which is ideal when many higher-order interaction terms are unphysical.
- **Ridge ($L_2$ penalty):** $\text{Loss} = \text{MSE} + \alpha \sum w_j^2$. The $L_2$ squared penalty shrinks all weights smoothly without setting them to zero. This keeps the curve stable and prevents coefficients from blowing up, which fits smooth spatial fields well.

---

## 4. Experimental Journey & Thought Process

### 4.1 Stage 1: Baseline Using PDF Hints (`model1_baseline.py`)

I started by directly implementing the starting hints from the problem description:
- `var1`: Degree 3 using $x_1, x_2, x_3$ with standard OLS.
- `var2`: Degree 4 using $x_1$ alone with standard OLS.

**Results:**
- `var1`: Train $R^2 = 0.2254$, 5-Fold CV $R^2 = \mathbf{0.1838}$, CV MSE = $\mathbf{8.9044}$
- `var2`: Train $R^2 = 0.1211$, 5-Fold CV $R^2 = \mathbf{0.1058}$, CV MSE = $\mathbf{36.678}$

**Why it failed:**
Both models underfitted heavily:
- For `var1`, discarding $x_4, x_5, x_6$ meant ignoring blade pitch and steam inlet pressure.
- For `var2`, using only $x_1$ ignored two of the three spatial dimensions. You cannot map a 3D subsurface temperature field from just an East-West coordinate.
- The takeaway was clear: because datasets are personalized, I needed to let cross-validation on the actual data guide model selection.

---

### 4.2 Stage 2: Systematic OLS Sweep (`model2_ols_sweep.py`)

Next, I ran a 5-fold cross-validation sweep over degrees 1 to 10 using all available features for both problems.

```
       var1 Validation MSE vs Degree                   var2 Validation MSE vs Degree
  MSE                                             MSE
 10.0 ┼  Deg 1 (9.76)                            35.0 ┼  Deg 1 (32.10)
  3.5 ┼        Deg 2 (3.51)                      19.8 ┼        Deg 2 (19.79)
  1.2 ┼              Deg 3 (1.16)                11.1 ┼              Deg 3 (11.13)
  0.8 ┼                    Deg 4 (0.81) ★ OLS     3.4 ┼                    Deg 4 (3.45)
  1.6 ┼                          Deg 5 (1.63)     0.6 ┼                          Deg 6 (0.62)
 97.3 ┼                                Deg 6      0.3 ┼                                Deg 8 (0.29) ★ OLS
      └──────────────────────────────────────         └──────────────────────────────────────
           1     2     3     4     5     6                 1     2     3     4     6     8
```

- **`var1` (All 6 Features):**
  - Degree 1: CV $R^2 = 0.1007$, CV MSE = $9.7633$
  - Degree 2: CV $R^2 = 0.6726$, CV MSE = $3.5145$
  - Degree 3: CV $R^2 = 0.8900$, CV MSE = $1.1650$
  - **Degree 4: CV $R^2 = \mathbf{0.9230}$, CV MSE = $\mathbf{0.8081}$ (OLS Peak)**
  - Degree 5: CV $R^2 = 0.8399$, CV MSE = $1.6288$ (error doubled as 462 terms started fitting noise)
  - Degree 6: CV $R^2 = -7.7534$, CV MSE = $97.3203$ (severe collapse)

- **`var2` (All 3 Features):**
  - Degree 1: CV $R^2 = 0.2182$, CV MSE = $32.1021$
  - Degree 4: CV $R^2 = 0.9142$, CV MSE = $3.4457$
  - Degree 6: CV $R^2 = 0.9848$, CV MSE = $0.6158$
  - **Degree 8: CV $R^2 = \mathbf{0.9927}$, CV MSE = $\mathbf{0.2896}$ (OLS Peak)**
  - Degree 9: CV $R^2 = 0.9916$, CV MSE = $0.3372$ (mild overfitting starts)

![Bias-Variance Tradeoff Curves](figures/bias_variance_tradeoff.png)
*Figure 1: Validation MSE across polynomial degrees. The U-shaped curves show where unregularized OLS begins to overfit, and mark the optimal regularized models.*

**Takeaway:**
Including all features gave a huge improvement ($R^2$ jumped from 0.18 to 0.92 on `var1`, and from 0.10 to 0.99 on `var2`). But OLS hit a ceiling: going past degree 4 on `var1` or degree 8 on `var2` led to overfitting.

---

### 4.3 Stage 3 & 3+: Regularization (`train_predict.py` & `model3_regularized.py`)

To evaluate higher degrees safely, I applied Lasso ($L_1$) and Ridge ($L_2$).

#### Why Lasso is best for `var1`
At degree 5, there are 462 polynomial features. When I fitted Lasso ($\alpha = 0.00348$), I inspected the weights:
- **Total polynomial features:** 462
- **Weights driven to exactly 0.0:** **364 terms (78.8%)**
- **Active weights kept:** **98 terms (21.2%)**

Lasso automatically pruned away almost 80% of the terms, keeping only the genuine physical interactions. Ridge kept all 462 terms non-zero and reached CV $R^2 = 0.9521$, whereas Lasso reached **CV $R^2 = \mathbf{0.9689}$** and lowered CV MSE to **$0.3218$**.

#### Why Ridge is best for `var2`
For `var2`, degree 9 produces 220 terms. Because subsurface temperature fields vary smoothly across continuous 3D space, zeroing out terms creates artificial bumps in the spatial surface. Ridge shrinks all 220 coefficients smoothly:
- Optimal $\alpha = 0.006649$
- Total $L_2$ weight norm: $38.94$ (maximum single weight: $7.43$)
- **CV $R^2 = \mathbf{0.9929}$**, **CV MSE = $\mathbf{0.2835}$**

![Sparsity and Coefficient Shrinkage](figures/sparsity_and_coefficients.png)
*Figure 2: Top: Lasso setting 364 terms to zero on var1 at degree 5. Bottom: Ridge keeping weights stable on var2 compared to the exploded weights of unconstrained OLS.*

---

### 4.4 Stage 4: Testing Overfitting (`model4_overfit.py`)

To see what happens when polynomial degree is pushed too far without regularization, I built two deliberately extreme models:
- `var1`: Degree 8 OLS on 6 features $\to$ **3,003 terms** (more terms than training rows!).
- `var2`: Degree 15 OLS on 3 features $\to$ **816 terms**.

**What happened:**
- On training data, both models fit almost perfectly (Train $R^2 \approx 0.9999$).
- But on validation folds, `var1` CV $R^2$ dropped to **0.6757** (MSE = $3.414$), and `var2` CV $R^2$ crashed to **-4.906** (MSE exploded to **285.50**).
- For `var2`, the unconstrained OLS weights blew up to over **$\pm 4,600$** with an $L_2$ norm of **$27,038$** (compared to $38.9$ with Ridge). The curve oscillated wildly between points (Runge's phenomenon).

This confirmed that Model 3 sits right at the optimal balance point.

---

## 5. Summary of Results

Here is the comparison across all four stages:

| Stage | Script | Problem | Features | Degree | Terms | Model | Train $R^2$ | Train MSE | CV $R^2$ | CV MSE | Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **M1** | `model1_baseline.py` | `var1` | $x_1-x_3$ | 3 | 20 | OLS | 0.2254 | 8.452 | $0.1838 \pm 0.038$ | $8.904 \pm 1.79$ | Heavy underfitting |
| **M1** | `model1_baseline.py` | `var2` | $x_1$ only | 4 | 5 | OLS | 0.1211 | 36.036 | $0.1058 \pm 0.021$ | $36.678 \pm 8.02$ | Heavy underfitting |
| **M2** | `model2_ols_sweep.py` | `var1` | $x_1-x_6$ | 4 | 210 | OLS | 0.9388 | 0.667 | $0.9230 \pm 0.010$ | $0.808 \pm 0.11$ | Solid baseline |
| **M2** | `model2_ols_sweep.py` | `var2` | $x_1-x_3$ | 8 | 165 | OLS | 0.9950 | 0.203 | $0.9927 \pm 0.001$ | $0.290 \pm 0.04$ | Solid baseline |
| **M3** | `model3_regularized.py`| `var1` | $x_1-x_6$ | 5 | 462 | Lasso ($\alpha \approx 0.0033$) | 0.9774 | 0.248 | $0.9686 \pm 0.006$ | $0.333 \pm 0.03$ | Sparsity helps |
| **M3** | `model3_regularized.py`| `var2` | $x_1-x_3$ | 9 | 220 | Ridge ($\alpha \approx 0.0066$) | 0.9959 | 0.168 | $0.9929 \pm 0.001$ | $0.284 \pm 0.04$ | Smooth shrinkage |
| **M3+**| **`train_predict.py`** | **`var1`** | **$x_1-x_6$** | **5** | **462** | **Lasso ($\alpha = 0.00348$)** | **0.9774** | **0.247** | **`0.9689 ± 0.005`**| **`0.322 ± 0.031`**| **Best model (Submit)** |
| **M3+**| **`train_predict.py`** | **`var2`** | **$x_1-x_3$** | **9** | **220** | **Ridge ($\alpha = 0.00665$)** | **0.9959** | **0.167** | **`0.9929 ± 0.001`**| **`0.284 ± 0.037`**| **Best model (Submit)** |
| **M4** | `model4_overfit.py` | `var1` | $x_1-x_6$ | 8 | 3,003 | OLS | 0.9998 | 0.002 | $0.6757 \pm 0.089$ | $3.414 \pm 0.94$ | High variance |
| **M4** | `model4_overfit.py` | `var2` | $x_1-x_3$ | 15 | 816 | OLS | 0.9999 | 0.0001 | $-4.906 \pm 3.820$ | $285.50 \pm 189.2$ | Severe overfitting |

![Model Comparison Across Stages](figures/model_comparison.png)
*Figure 3: Cross-validation R² and log MSE across all 4 stages, illustrating the improvement from M1 to M3+ and the failure of M4.*

---

## 6. Model Diagnostics & Prediction Checks

### 6.1 Residual Checks
I analyzed the out-of-fold residuals ($e_i = y_i - \hat{y}_i$):
- **`var1` (Degree 5 Lasso):** Mean residual is essentially zero ($-2.3 \times 10^{-16}$), standard deviation is $0.498$, and the distribution is symmetric and centered at zero.
- **`var2` (Degree 9 Ridge):** Mean residual is essentially zero ($+1.7 \times 10^{-15}$), standard deviation is $0.410$, and residuals are normally distributed without visible fan patterns (homoscedastic).

![Residual Diagnostics](figures/residual_diagnostics.png)
*Figure 4: Residual histograms with normal fit (top) and residuals vs. predicted values (bottom), showing zero-mean, symmetric errors.*

![Actual vs Predicted Out-of-Fold](figures/actual_vs_predicted.png)
*Figure 5: 5-Fold cross-validated predictions against ground truth along the 1:1 line ($y = \hat{y}$).*

### 6.2 Prediction File Verification
Final predictions from `train_predict.py` were checked against the submission instructions:
- Output files: `BT2024054/BT2024054_pred_var1.csv` and `BT2024054/BT2024054_pred_var2.csv`.
- Exactly 1,000 rows each, with a single column header `y` matching `sample_submission.csv`.
- No NaN, null, or infinite values.
- Prediction distributions closely match the training labels:
  - `var1` predictions: mean $1.28$, std $4.21$, range $[-9.77, 15.78]$ (train mean $0.76$, range $[-10.43, 11.48]$).
  - `var2` predictions: mean $1.88$, std $6.65$, range $[-29.53, 29.26]$ (train mean $2.40$, range $[-29.69, 39.25]$).

---

## 7. Key Takeaways

1. **Don't rely blindly on starting hints:** Problem descriptions often provide simple starting points. Cross-validation on the actual dataset showed that keeping all features and using higher degrees was essential.
2. **Missing features hurt the most:** No amount of hyperparameter tuning can make up for leaving out real physical variables.
3. **Choose regularization based on the physical problem:**
   - For multi-variable systems with many possible interactions (`var1`), **Lasso** removes unphysical cross-terms (pruning 78.8% of terms).
   - For continuous 3D spatial fields (`var2`), **Ridge** keeps the spatial surface smooth and prevents weight explosion.
4. **Overfitting is real at high degrees:** Model 4 showed that near-zero training error is easy to get with high degrees, but without regularization, predictions on unseen data can fail catastrophically.

---

## 8. Reproducibility & GitHub

All code, data, and predictions are available in the repository:

```bash
# Clone the repo
git clone https://github.com/Ayush-patel9/ML_ASSIGNMENT.git
cd ML_ASSIGNMENT

# Install requirements
pip install numpy pandas scikit-learn matplotlib

# Generate final predictions
python3 train_predict.py

# Recreate all figures in ./figures/
python3 generate_plots.py

# Run individual stages:
python3 model1_baseline.py       # Stage 1 (Baseline)
python3 model2_ols_sweep.py      # Stage 2 (OLS sweep)
python3 model3_regularized.py    # Stage 3 (Regularized)
python3 model4_overfit.py        # Stage 4 (Overfit demo)
```

- **Repository:** [https://github.com/Ayush-patel9/ML_ASSIGNMENT](https://github.com/Ayush-patel9/ML_ASSIGNMENT)
- **Student:** Ayush Patel (**BT2024054**)
