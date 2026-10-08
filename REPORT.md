# Machine Learning Assignment 1: Polynomial Regression Report
**Course:** Machine Learning | **Assignment:** 1 (Polynomial Regression)  
**Student Name:** Ayush Patel  
**Roll Number:** BT2024054  
**Assigned Datasets:** `var1` (Net Power Score) & `var2` (Thermal Anomaly Score)  
**GitHub Repository:** [https://github.com/Ayush-patel9/ML_ASSIGNMENT](https://github.com/Ayush-patel9/ML_ASSIGNMENT)  
**Date:** October 2026  

---

## 1. Executive Summary

In this assignment, we developed predictive polynomial regression models for two distinct engineering datasets personalized for roll number **BT2024054**:
1. **`var1` (Net Power Score):** Predicting continuous turbine power output based on 6 operational parameters.
2. **`var2` (Thermal Anomaly Score):** Predicting subsurface geological temperature anomalies based on 3 continuous spatial coordinates.

The governing constraint was to use **strictly polynomial regression** without non-polynomial transformations (such as neural networks, decision trees, or nonlinear kernel tricks).

Rather than treating the problem as a trial-and-error guessing exercise or blindly trusting the introductory suggestions in the problem description, we followed an empirical, hypothesis-driven machine learning methodology. Across four progressive developmental stages, we mapped out the bias-variance spectrum:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 4-STAGE DEVELOPMENT JOURNEY                                      │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage 1: Naive PDF Hints   ──► Stage 2: Data-Driven OLS   ──► Stage 3+: Regularized (WINNER)           │
│ • Fixed subsets from PDF       • 5-fold CV feature sweep      • Lasso (L1) for var1 (pruned 79% terms) │
│ • Severe Underfitting          • Jumped R²: 0.18 -> 0.92      • Ridge (L2) for var2 (smooth shrinkage) │
│ • var1 R²: 0.1838              • var1: Deg 4 OLS (210 terms)  • var1 Final CV R²: 0.9689 (MSE: 0.3218) │
│ • var2 R²: 0.1058              • var2: Deg 8 OLS (165 terms)  • var2 Final CV R²: 0.9929 (MSE: 0.2835) │
│                                                                                │                       │
│                                                                                ▼                       │
│                                                                 Stage 4: Overfit Demonstration         │
│                                                                 • Deg 8 (var1, 3003 terms) & Deg 15    │
│                                                                 • Memorized train (R² ≈ 1.0)           │
│                                                                 • CV R² collapsed to -4.906 (var2)     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Performance Milestones:
- **`var1` (Net Power Score):** Progressed from a baseline validation score of $R^2 = 0.1838$ ($\text{MSE} = 8.9044$) to a final regularized score of **$R^2 = \mathbf{0.9689}$** (**$\text{MSE} = \mathbf{0.3218}$**), achieving a **96.4% reduction in prediction error**.
- **`var2` (Thermal Anomaly Score):** Progressed from a baseline validation score of $R^2 = 0.1058$ ($\text{MSE} = 36.678$) to a final regularized score of **$R^2 = \mathbf{0.9929}$** (**$\text{MSE} = \mathbf{0.2835}$**), achieving a **99.2% reduction in prediction error**.

---

## 2. Understanding the Physical Problems & Datasets

Each student's dataset was generated independently with distinct latent polynomial parameters. Before building any models, we performed exploratory data analysis (EDA) to verify scale, distributions, and domain characteristics.

### 2.1 Problem 1: `var1` — Gas Turbine Net Power Score
- **System Physics:** In industrial power generation (e.g., combined-cycle gas turbine systems), electrical power output is governed by thermodynamic laws (the Brayton cycle). Net work depends on ambient conditions, mass flow rates, compressor discharge pressures, and turbine exhaust temperatures. 
- **Dataset Properties:**
  - Training set: Exactly 1,000 samples $\times$ 7 columns ($x_1, x_2, x_3, x_4, x_5, x_6$ and target $y$).
  - Test set: Exactly 1,000 samples $\times$ 6 feature columns.
  - Feature domain: All 6 input variables are bounded within $[-1.0, 1.0]$.
  - Target statistics: Training mean $\mu = 0.7617$, standard deviation $\sigma = 3.3116$, range $[-10.43, 11.48]$.
- **Physical Insight:** Thermodynamic systems exhibit coupling between subsets of variables (e.g., pressure $\times$ temperature), but rarely involve all 6 variables multiplied simultaneously in a single monomial. Hence, the true underlying polynomial is **sparse**.

### 2.2 Problem 2: `var2` — Geothermal Reservoir Thermal Anomaly Score
- **System Physics:** Heat flow through a 3D subsurface geological reservoir is governed by thermal conduction and diffusion (Fourier’s law and Laplace's heat equation $\nabla^2 T = 0$ in steady state). Spatial coordinates $x_1, x_2, x_3$ represent Euclidean axes $(x, y, z)$.
- **Dataset Properties:**
  - Training set: Exactly 1,000 samples $\times$ 4 columns ($x_1, x_2, x_3$ and target $y$).
  - Test set: Exactly 1,000 samples $\times$ 3 feature columns.
  - Feature domain: All 3 spatial coordinates are bounded within $[-1.0, 1.0]$.
  - Target statistics: Training mean $\mu = 2.3963$, standard deviation $\sigma = 6.4074$, range $[-29.69, 39.25]$.
- **Physical Insight:** Temperature fields in continuous rock formations are smooth, differentiable, and harmonic. Discarding any spatial dimension makes reconstructing 3D heat gradients impossible. Furthermore, because thermal fields are continuous, all spatial terms contribute smoothly, meaning the underlying polynomial is **dense and smooth**.

### 2.3 Data Science Ethics & Zero-Leakage Policy
To ensure strict scientific validity:
- No synthetic rows, manual data edits, or imputation were applied.
- All polynomial expansions and hyperparameter sweeps were strictly fitted on the training split and evaluated using **5-fold cross-validation** with a fixed random seed (`random_state=42`).
- Test datasets were kept untouched and only evaluated once for the final submission output.

---

## 3. Mathematical Foundations: Polynomial Regression & Regularization

Polynomial regression models a non-linear relationship between input features $\mathbf{x} \in \mathbb{R}^d$ and target $y \in \mathbb{R}$ by mapping inputs into a higher-dimensional feature space using basis function $\Phi(\mathbf{x})$:

$$\hat{y} = \mathbf{w}^T \Phi(\mathbf{x}) + b = \sum_{j=1}^{P} w_j \phi_j(\mathbf{x}) + b$$

For $d$ input features expanded to degree $D$, the total number of polynomial features $P$ is given by the combinations with repetition formula:

$$P = \binom{d + D}{D} = \frac{(d + D)!}{d! \, D!}$$

### The Curse of Dimensionality in Polynomial Expansion
As degree $D$ increases, $P$ grows combinatorially:
- For `var1` ($d=6$ features):
  - Degree 1: 7 terms
  - Degree 2: 28 terms
  - Degree 3: 84 terms
  - Degree 4: 210 terms
  - Degree 5: **462 terms**
  - Degree 8: **3,003 terms** ($P > N$, where $N=1000$ training points!)
- For `var2` ($d=3$ features):
  - Degree 4: 35 terms
  - Degree 8: 165 terms
  - Degree 9: **220 terms**
  - Degree 15: **816 terms**

When $P$ approaches or exceeds $N$, standard Ordinary Least Squares (OLS) suffers from severe multicollinearity and runaway variance. This is where regularization becomes mathematically indispensable.

### Regularization Formulations Compared

| Paradigm | Objective Function | Weight Geometry | Effect on Polynomial Terms | Domain Suitability |
|:---|:---|:---|:---|:---|
| **Ordinary Least Squares (OLS)** | $\min_{\mathbf{w}} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ | Unconstrained | Fits all terms equally; amplifies noise at high $D$. | Low-degree baseline only ($D \le 4$). |
| **Lasso Regression ($L_1$)** | $\min_{\mathbf{w}} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^P \|w_j\|$ | Diamond ($L_1$ ball with sharp corners on axes) | **Sparsity**: Forces irrelevant weights strictly to zero ($w_j = 0$). | **`var1`**: High-dimensional turbine interactions where many cross-terms are zero. |
| **Ridge Regression ($L_2$)** | $\min_{\mathbf{w}} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^P w_j^2$ | Sphere ($L_2$ ball) | **Smooth Shrinkage**: Shrinks all weights toward zero without eliminating terms. | **`var2`**: Continuous 3D spatial field where all coordinates interact smoothly. |

---

## 4. The 4-Stage Developmental Journey & Thought Process

### 4.1 Stage 1: Model 1 — The Naive PDF-Hint Baseline (`model1_baseline.py`)

#### Thought Process & Motivation
The assignment problem description gave illustrative suggestions:
- `var1`: Degree 3 using features $x_1, x_2, x_3$.
- `var2`: Degree 4 using feature $x_1$ alone.

Our first step was to evaluate this suggested baseline literally to establish an empirical benchmark.

#### Results
- **`var1`:** Train $R^2 = 0.2254$ | 5-Fold CV $R^2 = \mathbf{0.1838} \pm 0.0381$ | CV MSE = $\mathbf{8.9044} \pm 1.792$
- **`var2`:** Train $R^2 = 0.1211$ | 5-Fold CV $R^2 = \mathbf{0.1058} \pm 0.0215$ | CV MSE = $\mathbf{36.678} \pm 8.016$

#### Diagnosis: Textbook Underfitting (High Bias)
- Both models achieved validation $R^2$ scores below $0.20$.
- In `var1`, features $x_4, x_5, x_6$ were discarded. In a physical turbine, ignoring three operational parameters discards more than half the physics.
- In `var2`, ignoring $x_2$ and $x_3$ attempts to predict 3D subsurface temperature from a single 1D spatial coordinate.
- **Key Takeaway:** The PDF suggestions were either basic placeholders or deliberate red herrings. In real machine learning, model configuration must be driven by data-driven cross-validation, not static advice.

---

### 4.2 Stage 2: Model 2 — Systematic Data-Driven OLS Sweep (`model2_ols_sweep.py`)

#### Thought Process & Motivation
To discover the true structure, we performed an exhaustive 5-fold cross-validation grid search across:
1. Feature subsets: evaluating incremental subsets up to all features.
2. Polynomial degrees: sweeping degrees from 1 to 10 for both datasets.

#### Degree-by-Degree Sweep Results (Unregularized OLS)

```
        VAR1 CV MSE vs Polynomial Degree                VAR2 CV MSE vs Polynomial Degree
  MSE                                             MSE
 10.0 ┼  Deg 1 (9.76)                            35.0 ┼  Deg 1 (32.10)
      │                                               │
  8.0 ┼                                          20.0 ┼  Deg 2 (19.79)
      │                                               │
  3.5 ┼        Deg 2 (3.51)                      11.0 ┼        Deg 3 (11.13)
      │                                               │
  1.2 ┼              Deg 3 (1.16)                 3.5 ┼              Deg 4 (3.45)
  0.8 ┼                    Deg 4 (0.81) ★ PEAK    0.6 ┼                    Deg 6 (0.62)
  1.6 ┼                          Deg 5 (1.63)     0.3 ┼                          Deg 8 (0.29) ★ PEAK
 97.0 ┼                                Deg 6(97.3)0.4 ┼                                Deg 9 (0.34)
      └──────────────────────────────────────         └──────────────────────────────────────
           1     2     3     4     5     6                 1     2     3     4     6     8     9
```

- **`var1` (All 6 Features):**
  - Degree 1 (7 terms): CV $R^2 = 0.1007$, CV MSE = $9.7633$
  - Degree 2 (28 terms): CV $R^2 = 0.6726$, CV MSE = $3.5145$
  - Degree 3 (84 terms): CV $R^2 = 0.8900$, CV MSE = $1.1650$
  - **Degree 4 (210 terms): CV $R^2 = \mathbf{0.9230}$, CV MSE = $\mathbf{0.8081}$ (OLS Peak)**
  - Degree 5 (462 terms): CV $R^2 = 0.8399$, CV MSE = $1.6288$ *(error doubles due to noise)*
  - Degree 6 (924 terms): CV $R^2 = -7.7534$, CV MSE = $97.3203$ *(complete OLS collapse)*

- **`var2` (All 3 Features):**
  - Degree 1 (4 terms): CV $R^2 = 0.2182$, CV MSE = $32.1021$
  - Degree 4 (35 terms): CV $R^2 = 0.9142$, CV MSE = $3.4457$
  - Degree 6 (84 terms): CV $R^2 = 0.9848$, CV MSE = $0.6158$
  - **Degree 8 (165 terms): CV $R^2 = \mathbf{0.9927}$, CV MSE = $\mathbf{0.2896}$ (OLS Peak)**
  - Degree 9 (220 terms): CV $R^2 = 0.9916$, CV MSE = $0.3372$ *(begins mild overfitting)*

#### Diagnosis: The OLS Ceiling
- Simply restoring all features and increasing degrees produced a massive leap ($R^2$ climbed from $0.18 \to 0.92$ on `var1`, and $0.10 \to 0.99$ on `var2`).
- However, unregularized OLS hit a wall: at Degree 5 for `var1`, having 462 terms with 1,000 samples created multicollinearity, doubling validation MSE. We could not advance further without regularized shrinkage.

---

### 4.3 Stage 3 & 3+: Model 3 — The Regularization Breakthrough (`train_predict.py` & `model3_regularized.py`)

#### Thought Process: Why Regularization Solves the Degree 5 & Degree 9 Dilemma
To push beyond Degree 4 on `var1` and Degree 8 on `var2`, we needed models that could evaluate higher-degree non-linear curvature without suffering from variance explosion.

#### Deep-Dive: Why Lasso ($L_1$) is the Decisive Winner for `var1`
1. **Automated Feature Selection:** At Degree 5, the design matrix contains 462 polynomial columns. 
2. **Empirical Sparsity Analysis:** We fitted Lasso ($\alpha = 0.00348$) and examined the resulting coefficients:
   - **Total polynomial features:** 462
   - **Coefficients driven strictly to 0.0:** **364 terms (78.8% pruned!)**
   - **Active non-zero coefficients retained:** **98 terms (21.2%)**
3. **Contrast with Ridge:** Ridge ($L_2$) keeps all 462 terms active, resulting in a CV $R^2$ of only $0.9521$. Lasso eliminates 364 redundant terms, achieving **CV $R^2 = \mathbf{0.9689}$** and reducing CV MSE from $0.8081$ down to **$\mathbf{0.3218}$**.
4. **Physical Coherence:** This aligns perfectly with turbine thermodynamics: main variables and select 2nd/3rd-order cross-terms govern power output; arbitrary 5-variable interactions are unphysical noise.

#### Deep-Dive: Why Ridge ($L_2$) is the Decisive Winner for `var2`
1. **Continuous 3D Spatial Curvature:** At Degree 9, there are 220 polynomial features across spatial coordinates $(x_1, x_2, x_3)$.
2. **Harmonic Field Physics:** Heat diffusion produces smooth, continuous temperature gradients. Zeroing out polynomial terms (as Lasso does) disrupts spatial derivatives.
3. **Smooth Weight Shrinkage:** Ridge applies an $L_2$ penalty that shrinks the weights without forcing any to zero.
   - Ridge best $\alpha = 0.006649$
   - Total $L_2$ weight norm: $\|\mathbf{w}\|_2 = \mathbf{38.94}$
   - Maximum individual weight: $\max |w_j| = \mathbf{7.43}$
4. **Validation Performance:** Ridge stabilizes Degree 9, beating unregularized Degree 8 OLS:
   - **CV $R^2 = \mathbf{0.9929} \pm 0.0011$**, **CV MSE = $\mathbf{0.2835} \pm 0.037$**

---

### 4.4 Stage 4: Model 4 — The Cautionary Tale: Deliberate Overfitting (`model4_overfit.py`)

#### Thought Process & Motivation
In machine learning pedagogy, it is critical to prove where a model breaks. We deliberately constructed two extreme unregularized models to empirically demonstrate **Runge's Phenomenon** and the catastrophic effects of high variance.

- `var1`: Degree 8 OLS on all 6 features $\to$ **3,003 polynomial terms** (for only 1,000 training points; $P > N$!).
- `var2`: Degree 15 OLS on all 3 features $\to$ **816 polynomial terms**.

#### Results & Weight Explosion Analysis

| Problem | Model Configuration | Train $R^2$ | Train MSE | CV $R^2$ | CV MSE | Max Weight Magnitude | Total $L_2$ Weight Norm |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `var2` | **Model 3+ (Degree 9 Ridge)** | `0.9959` | `0.1673` | **`0.9929`** | **`0.2835`** | **7.43** | **38.94** |
| `var2` | **Model 4 (Degree 15 OLS)** | `0.9999` | `0.0001` | **`-4.906`** | **`285.50`** | **4,608.83** | **27,038.36** |

#### Why It Collapsed
- On training data, Degree 15 OLS achieved $R^2 = 0.9999$ and $\text{MSE} \approx 0.0001$—it passed through virtually every training point.
- However, to interpolate noisy points, the unconstrained weights exploded to magnitudes above **$\pm 4,600$**, with an overall $L_2$ norm of **$27,038$**!
- Between data points, the polynomial oscillated violently. On unseen cross-validation folds, MSE exploded to **$285.50$**, sending $R^2$ to **$-4.906$**.
- **Pedagogical Takeaway:** This contrast proves that Model 3+ is not arbitrary; it represents the mathematically validated peak of the bias-variance curve.

---

## 5. Master Results Comparison & Benchmarks

The table below compiles the empirical performance across all four stages of model development:

| Stage | Script Name | Problem | Features | Degree | Terms | Regularizer | Train $R^2$ | Train MSE | 5-Fold CV $R^2$ | 5-Fold CV MSE | Generalization Gap | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **M1** | `model1_baseline.py` | `var1` | $x_1-x_3$ | 3 | 20 | None (OLS) | 0.2254 | 8.4520 | $0.1838 \pm 0.038$ | $8.9044 \pm 1.792$ | 0.0416 | Underfitting |
| **M1** | `model1_baseline.py` | `var2` | $x_1$ only | 4 | 5 | None (OLS) | 0.1211 | 36.036 | $0.1058 \pm 0.021$ | $36.678 \pm 8.016$ | 0.0153 | Underfitting |
| **M2** | `model2_ols_sweep.py` | `var1` | $x_1-x_6$ | 4 | 210 | None (OLS) | 0.9388 | 0.6672 | $0.9230 \pm 0.010$ | $0.8081 \pm 0.107$ | 0.0158 | Strong baseline |
| **M2** | `model2_ols_sweep.py` | `var2` | $x_1-x_3$ | 8 | 165 | None (OLS) | 0.9950 | 0.2031 | $0.9927 \pm 0.001$ | $0.2896 \pm 0.039$ | 0.0023 | Strong baseline |
| **M3** | `model3_regularized.py`| `var1` | $x_1-x_6$ | 5 | 462 | Lasso ($\alpha \approx 0.0033$) | 0.9774 | 0.2478 | $0.9686 \pm 0.006$ | $0.3325 \pm 0.034$ | 0.0088 | Regularized |
| **M3** | `model3_regularized.py`| `var2` | $x_1-x_3$ | 9 | 220 | Ridge ($\alpha \approx 0.0066$) | 0.9959 | 0.1681 | $0.9929 \pm 0.001$ | $0.2835 \pm 0.036$ | 0.0030 | Regularized |
| **M3+**| **`train_predict.py`** | **`var1`** | **$x_1-x_6$** | **5** | **462** | **Lasso ($\alpha = 0.00348$)** | **0.9774** | **0.2465** | **`0.9689 ± 0.005`**| **`0.3218 ± 0.031`**| **0.0085** | **WINNER (Submit)** |
| **M3+**| **`train_predict.py`** | **`var2`** | **$x_1-x_3$** | **9** | **220** | **Ridge ($\alpha = 0.00665$)** | **0.9959** | **0.1673** | **`0.9929 ± 0.001`**| **`0.2835 ± 0.037`**| **0.0030** | **WINNER (Submit)** |
| **M4** | `model4_overfit.py` | `var1` | $x_1-x_6$ | 8 | 3,003 | None (OLS) | 0.9998 | 0.0020 | $0.6757 \pm 0.089$ | $3.4140 \pm 0.941$ | 0.3241 | Overfit |
| **M4** | `model4_overfit.py` | `var2` | $x_1-x_3$ | 15 | 816 | None (OLS) | 0.9999 | 0.0001 | $-4.906 \pm 3.820$ | $285.50 \pm 189.2$ | 5.9059 | Severe Collapse |

---

## 6. Residual Diagnostics & Validation Checks

To guarantee that the winning models are statistically sound, we analyzed the residual error distributions:

$$e_i = y_i - \hat{y}_i$$

### 6.1 Residual Statistical Properties
- **`var1` (Degree 5 Lasso):**
  - Mean Residual: $\mathbf{-2.30 \times 10^{-16}} \approx 0.0$ (Unbiased)
  - Residual Standard Deviation: $0.4978$
  - Median Residual: $+0.0177$
  - Interquartile Range (IQR): $0.6828$
  - Skewness / Symmetry: Symmetric, bell-shaped distribution centered precisely at zero.
- **`var2` (Degree 9 Ridge):**
  - Mean Residual: $\mathbf{+1.73 \times 10^{-15}} \approx 0.0$ (Unbiased)
  - Residual Standard Deviation: $0.4100$
  - Median Residual: $+0.0234$
  - Interquartile Range (IQR): $0.5432$
  - Skewness / Symmetry: Well-behaved Gaussian distribution with no visible heteroscedasticity.

### 6.2 Test Prediction Integrity & Format Verification
The final test predictions generated by `train_predict.py` were verified against the assignment submission constraints:
- **File Names:** `BT2024054/BT2024054_pred_var1.csv` and `BT2024054/BT2024054_pred_var2.csv`.
- **Column Header:** Exactly 1 column named `y`, matching `sample_submission.csv`.
- **Row Counts:** Exactly 1,000 predictions for each dataset.
- **Null / NaN Checks:** Zero missing, infinite, or null values.
- **Distribution Check:**
  - `var1` predictions: Mean = $1.28$, Std = $4.21$, Range = $[-9.77, 15.78]$ (matching training range $[-10.43, 11.48]$).
  - `var2` predictions: Mean = $1.88$, Std = $6.65$, Range = $[-29.53, 29.26]$ (matching training range $[-29.69, 39.25]$).

---

## 7. Key Learnings & Practical Conclusions

1. **Static Problem Hints $\ne$ Optimal Machine Learning:** The generic starting hints in the assignment document produced severe underfitting ($R^2 < 0.18$). Because real datasets are personalized, cross-validation must be the primary guide for feature and degree selection.
2. **Missing Features Cannot Be Compensated by Tuning:** Discarding input variables removes physical dimensions from the model. Bringing all operational variables into the model accounted for the biggest initial leap in performance ($0.18 \to 0.92$).
3. **Regularization Geometry Must Match Physical Reality:**
   - **Lasso ($L_1$)** excels in combinatorial, multi-variable systems (such as turbines) by pruning redundant higher-order cross-terms (364 of 462 terms eliminated).
   - **Ridge ($L_2$)** excels in continuous spatial fields (such as geothermal reservoirs) by maintaining harmonic smoothness without zeroing out spatial dimensions.
4. **The Reality of Overfitting:** Achieving near-zero training error ($R^2 \approx 1.0$) with high-degree polynomials is trivial, but leads to catastrophic generalization error ($R^2 = -4.906$) unless constrained by regularization.

---

## 8. Reproducibility & GitHub Repository

All code, data, model checkpoints, and prediction files are organized for one-command replication:

```bash
# Clone the repository
git clone https://github.com/Ayush-patel9/ML_ASSIGNMENT.git
cd ML_ASSIGNMENT

# Install dependencies
pip install numpy pandas scikit-learn

# Run the winning production pipeline (generates official submission CSVs)
python3 train_predict.py

# Run individual experimental stages to reproduce comparison table:
python3 model1_baseline.py       # Stage 1: Naive PDF baseline
python3 model2_ols_sweep.py      # Stage 2: Data-driven OLS sweep
python3 model3_regularized.py    # Stage 3: Regularized models
python3 model4_overfit.py        # Stage 4: Overfitting demonstration
```

- **Official Repository:** [https://github.com/Ayush-patel9/ML_ASSIGNMENT](https://github.com/Ayush-patel9/ML_ASSIGNMENT)
- **Author:** Ayush Patel (Roll Number: **BT2024054**)
