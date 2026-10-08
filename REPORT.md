# Machine Learning Assignment 1: Polynomial Regression Report
**Student Name:** Ayush Patel  
**Roll Number:** BT2024054  
**Problem Identifiers:** `var1` (Net Power Score) & `var2` (Thermal Anomaly Score)  
**Course:** Machine Learning (Assignment 1)  
**Date:** October 2026  
**Repository:** [https://github.com/Ayush-patel9/ML_ASSIGNMENT](https://github.com/Ayush-patel9/ML_ASSIGNMENT)  

---

## 1. Executive Summary

In this assignment, we were tasked with predicting continuous numerical values for two personalized engineering datasets (`var1` and `var2`) assigned to roll number **BT2024054** using **strictly polynomial regression**.

Rather than treating machine learning as a guessing game or blindly accepting the initial suggestions in the assignment prompt, we adopted a systematic, hypothesis-driven approach. Over four progressive stages of experimentation, we analyzed the physics of each dataset, evaluated the bias-variance tradeoff, and identified the optimal polynomial representations:

1. **Model 1 (Naive Baseline):** Followed the PDF suggestions literally (Degree 3 on $x_1-x_3$ for `var1`, Degree 4 on $x_1$ only for `var2`). This suffered from severe underfitting ($R^2 = 0.1838$ and $0.1058$).
2. **Model 2 (Data-Driven OLS Sweep):** Performed exhaustive cross-validation sweeps across all feature subsets and degrees. Discovered that all features are essential, leaping performance to $R^2 = 0.9230$ and $0.9927$.
3. **Model 3 & 3+ (Regularized Models — Optimal Solution):** Introduced $L_1$ (Lasso) and $L_2$ (Ridge) regularization to control the explosion of polynomial terms at higher degrees. By matching the regularization geometry to the underlying physical system (Lasso for high-dimensional turbine interactions, Ridge for continuous 3D geological fields) and fine-tuning the penalty parameter $\alpha$, we achieved peak generalization:
   - **`var1` Final:** **$\text{CV } R^2 = 0.9689$** | **$\text{CV MSE} = 0.3218$** (Degree 5 + Lasso, $\alpha = 0.00348$)
   - **`var2` Final:** **$\text{CV } R^2 = 0.9929$** | **$\text{CV MSE} = 0.2835$** (Degree 9 + Ridge, $\alpha = 0.00665$)
4. **Model 4 (Extreme Overfitting Demonstration):** Deliberately pushed polynomial degrees to extremes without regularization to observe the variance explosion. While training error reached near zero ($R^2 \approx 1.0$), validation performance collapsed catastrophically ($R^2 = -4.906$ for `var2`), providing empirical proof of the sweet spot occupied by Model 3.

---

## 2. Understanding the Problems & Datasets

Each student received two independently generated datasets. Before building models, we performed exploratory data analysis to understand the structure, scale, and domain of each task.

### 2.1 Problem 1: `var1` — Net Power Score
- **Domain:** Performance modeling of a power generation turbine.
- **Inputs:** 6 operational variables ($x_1, x_2, x_3, x_4, x_5, x_6$).
- **Dataset Sizes:** 1,000 training instances, 1,000 test instances.
- **Input Properties:** All input variables are normalized and cleanly bounded within the range $[-1.0, 1.0]$.
- **Target Variable ($y$):** Continuous Net Power Score, with a training mean of $\approx 0.76$, standard deviation of $\approx 3.31$, spanning $[-10.43, 11.48]$.
- **Physical Context:** Turbine power output depends on non-linear thermodynamic interactions between flow rates, pressures, and temperatures. Discarding any input variable means leaving out a fundamental component of the physical system.

### 2.2 Problem 2: `var2` — Thermal Anomaly Score
- **Domain:** Subterranean geological reservoir temperature variation.
- **Inputs:** 3 continuous spatial coordinates ($x_1, x_2, x_3$) representing 3D coordinates $(x, y, z)$.
- **Dataset Sizes:** 1,000 training instances, 1,000 test instances.
- **Input Properties:** Bounded cleanly within $[-1.0, 1.0]$.
- **Target Variable ($y$):** Continuous Thermal Anomaly Score, with a training mean of $\approx 2.40$, standard deviation of $\approx 6.41$, spanning $[-29.69, 39.25]$.
- **Physical Context:** Temperature variations in a continuous 3D geological reservoir follow smooth spatial physics governed by the heat diffusion equation. Temperature at any coordinate $(x, y, z)$ depends simultaneously on all three spatial axes.

### 2.3 Experimental Integrity
Throughout all experiments, strict data science protocols were upheld:
- Training datasets were used strictly as provided; no manual editing, data leakage, or synthetic data augmentation was performed.
- All model evaluations used **5-fold cross-validation** (with random state fixed at 42 for exact reproducibility).
- Final test predictions were generated strictly from models fitted on the training split, completely unseen to the evaluation pipeline.

---

## 3. The Four-Stage Thought Process & Methodology

```
+---------------------------------------------------------------------------------------------------------+
|                                    EXPERIMENTAL PROGRESSION PIPELINE                                     |
+---------------------------------------------------------------------------------------------------------+
| Stage 1: Baseline (PDF Hints)   -->  Stage 2: OLS Exhaustive Sweep  -->  Stage 3: Regularized Models    |
| • Blind adherence to prompt          • Data-driven feature search        • Lasso (L1) for var1          |
| • OLS, Deg 3 (var1), Deg 4 (var2)    • OLS, Deg 4 (var1), Deg 8 (var2)   • Ridge (L2) for var2          |
| • R²: 0.1838 (v1), 0.1058 (v2)       • R²: 0.9230 (v1), 0.9927 (v2)      • R²: 0.9689 (v1), 0.9929 (v2) |
| [Severe Underfitting]                 [Unregularized Ceiling]             [Optimal Generalization]      |
|                                                                                         |               |
|                                                                                         v               |
|                                                                          Stage 4: Overfit Demo (OLS)    |
|                                                                          • Deg 8 (var1), Deg 15 (var2)  |
|                                                                          • R²: 0.6757 (v1), -4.906 (v2) |
|                                                                          [Catastrophic Variance]        |
+---------------------------------------------------------------------------------------------------------+
```

---

### 3.1 Stage 1: Model 1 — The Naive Starting Point (PDF Hints)

#### What We Did
The initial assignment document suggested starting points:
- For `var1`: A polynomial of degree 3 using only the first 3 features ($x_1, x_2, x_3$).
- For `var2`: A polynomial of degree 4 using only a single feature ($x_1$).

We implemented this directly in `model1_baseline.py` using standard Ordinary Least Squares (OLS) regression.

#### What Happened & Results
- **`var1`:** 5-fold CV $R^2 = \mathbf{0.1838} \pm 0.0381$, CV MSE = $\mathbf{8.9044} \pm 1.792$ (Train $R^2 = 0.2254$)
- **`var2`:** 5-fold CV $R^2 = \mathbf{0.1058} \pm 0.0215$, CV MSE = $\mathbf{36.678} \pm 8.016$ (Train $R^2 = 0.1211$)

#### Why It Failed (Thought Process & Diagnosis)
Both models exhibited textbook **high bias (underfitting)**:
1. **Discarding Active Features:** In `var1`, features $x_4, x_5, x_6$ were ignored. In a turbine, if you omit fuel pressure or exhaust temperature, you discard half the explanatory signal. In `var2`, predicting a 3D subsurface temperature while ignoring 2 of the 3 spatial coordinates ($x_2$ and $x_3$) makes accurate prediction impossible.
2. **Degree Too Low:** A degree 3 or 4 curve lacks the flexibility to express the true curvature present in the data.
3. **Lesson Learned:** The assignment hints were either broad generic starting points or deliberate distractions. Because datasets are personalized per student, true model selection must be driven by data and cross-validation, not static advice.

---

### 3.2 Stage 2: Model 2 — Systematic Data-Driven OLS Sweep

#### What We Did
In `model2_ols_sweep.py`, we removed all preconceptions and performed an exhaustive 5-fold cross-validation grid search across:
- **Feature Subsets:** Comparing models with $[x_1..x_3]$, $[x_1..x_4]$, $[x_1..x_5]$, and $[x_1..x_6]$ for `var1`; and $[x_1]$, $[x_1, x_2]$, and $[x_1, x_2, x_3]$ for `var2`.
- **Polynomial Degrees:** Sweeping degrees from 1 to 10 for both datasets.

#### What Happened & Results
- **`var1`:** Using all 6 features at **Degree 4** produced **CV $R^2 = 0.9230 \pm 0.0102$** and **CV MSE = $0.8081 \pm 0.107$** (210 polynomial terms).
- **`var2`:** Using all 3 spatial features at **Degree 8** produced **CV $R^2 = 0.9927 \pm 0.0011$** and **CV MSE = $0.2896 \pm 0.039$** (165 polynomial terms).

#### The Breakthrough & The Limitation
- **The Breakthrough:** By including all input features, the validation score skyrocketed ($0.18 \to 0.92$ on `var1`; $0.10 \to 0.99$ on `var2`). This confirmed beyond doubt that all inputs are active contributors to the target variables.
- **The OLS Ceiling:** When we attempted to increase `var1` to Degree 5 using unregularized OLS, the polynomial feature count grew to **462 terms**. With only 1,001 training rows, OLS started fitting sample-specific noise: CV $R^2$ dropped from 0.923 back down to 0.840, and CV MSE doubled to 1.629. Plain OLS could not climb higher without regularized constraints.

---

### 3.3 Stage 3: Model 3 & Tuned 3+ — Regularized Polynomial Regression (The Winning Architecture)

To push past the unregularized ceiling, we introduced mathematical regularization penalties. Regularization adds a penalty term to the loss function that discourages excessively large or unnecessary weights.

We evaluated two distinct regularization paradigms:
1. **Lasso ($L_1$ penalty):** $\text{Loss} = \text{MSE} + \alpha \sum_{j=1}^{p} |\beta_j|$
2. **Ridge ($L_2$ penalty):** $\text{Loss} = \text{MSE} + \alpha \sum_{j=1}^{p} \beta_j^2$

#### Why Lasso ($L_1$) is the Perfect Choice for `var1` (Net Power Score)
- **High Combinatorial Dimension:** Expanding 6 features to Degree 5 produces $\binom{6+5}{5} = 462$ polynomial terms.
- **Sparse Physical Reality:** In an operating turbine, physical laws are dominated by main effects and select pairwise/triplet interactions. The vast majority of 4-way and 5-way cross-product monomials do not correspond to real physical interactions.
- **Automatic Feature Selection:** The geometry of the $L_1$ diamond constraint has sharp corners on the coordinate axes. When the loss function touches these corners, it drives redundant coefficients **strictly to zero**. Lasso automatically pruned the 462 polynomial features down to only the essential subset.
- **Empirical Confirmation:** Ridge ($L_2$), which keeps all 462 coefficients non-zero, achieved only $R^2 = 0.9521$. Lasso reached **$0.9686$**.
- **Hyperparameter Fine-Tuning (Model 3+):** By zooming in on the regularization parameter $\alpha$ using high-resolution cross-validation around $\alpha = 0.00348$, we obtained our peak performance:
  $$\text{CV } R^2 = \mathbf{0.9689 \pm 0.0055}, \quad \text{CV MSE} = \mathbf{0.3218 \pm 0.031}$$

#### Why Ridge ($L_2$) is the Perfect Choice for `var2` (Thermal Anomaly Score)
- **Continuous 3D Geometry:** There are only 3 spatial inputs ($x_1, x_2, x_3$). Expanding to Degree 9 produces $\binom{3+9}{9} = 220$ polynomial terms.
- **Smooth Spatial Physics:** Geothermal temperature variation across continuous rock space obeys heat diffusion principles. Because heat diffuses smoothly in all directions, temperature gradients exist along all three spatial axes simultaneously.
- **Continuous Shrinkage:** Zeroing out polynomial terms (as Lasso does) introduces artificial spatial discontinuities. Ridge's circular $L_2$ constraint smoothly penalizes large weights without setting them to zero. This dampens high-frequency oscillations while preserving all spatial harmonics.
- **Empirical Confirmation:** Ridge at Degree 9 achieved:
  $$\text{CV } R^2 = \mathbf{0.9929 \pm 0.0011}, \quad \text{CV MSE} = \mathbf{0.2835 \pm 0.037}$$
  with an optimal penalty parameter $\alpha \approx 0.00665$.

---

### 3.4 Stage 4: Model 4 — The Cautionary Tale: Deliberate Overfitting

#### What We Did
In machine learning, it is easy to assume that higher polynomial degrees always yield better fits. To empirically test this boundary and complete our investigation of the bias-variance curve, we wrote `model4_overfit.py`:
- `var1`: Pushed to **Degree 8 OLS** on all 6 features $\to$ **3,003 polynomial terms** (for only 1,001 training samples!).
- `var2`: Pushed to **Degree 15 OLS** on all 3 features $\to$ **816 polynomial terms**.

#### What Happened & Results
- **`var1` Overfit:**
  - Training Fit: $R^2 = \mathbf{0.9998}$, $\text{MSE} = \mathbf{0.0020}$ (Virtually zero error on seen data!)
  - 5-Fold Validation: CV $R^2 = \mathbf{0.6757}$, CV MSE = $\mathbf{3.414}$
- **`var2` Overfit:**
  - Training Fit: $R^2 = \mathbf{0.9999}$, $\text{MSE} = \mathbf{0.0001}$ (Near-perfect interpolation)
  - 5-Fold Validation: CV $R^2 = \mathbf{-4.906}$, CV MSE = $\mathbf{285.50}$ (Complete generalization collapse!)

#### Why This Matters (The Bias-Variance Tradeoff in Action)
Model 4 provides a textbook demonstration of **Runge's Phenomenon**:
- When an unconstrained polynomial of high degree is fitted to data, the curve violently oscillates between data points.
- On the training data, the polynomial passes through virtually every point. But on unseen validation data, these oscillations produce catastrophic errors—pushing MSE from $0.28$ all the way to $285.50$, and sending $R^2$ deep into negative territory.
- This contrast proves why **Model 3+** is the scientifically sound choice: it captures non-linear curvature without succumbing to the runaway variance exhibited in Model 4.

---

## 4. Comprehensive Experimental Results

The table below compiles the performance metrics across all four experimental stages for both tasks:

| Model Stage | Problem | Features Used | Poly Degree | Total Poly Terms | Regularizer | Train $R^2$ | Train MSE | 5-Fold CV $R^2$ | 5-Fold CV MSE | Evaluation Summary |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Model 1 (Baseline)** | `var1` | $x_1 - x_3$ | 3 | 20 | None (OLS) | 0.2254 | 8.452 | $0.1838 \pm 0.038$ | $8.9044 \pm 1.79$ | Severe Underfitting |
| **Model 1 (Baseline)** | `var2` | $x_1$ only | 4 | 5 | None (OLS) | 0.1211 | 36.036 | $0.1058 \pm 0.021$ | $36.678 \pm 8.02$ | Severe Underfitting |
| **Model 2 (OLS Sweep)** | `var1` | $x_1 - x_6$ | 4 | 210 | None (OLS) | 0.9388 | 0.667 | $0.9230 \pm 0.010$ | $0.8081 \pm 0.11$ | Good; hits OLS ceiling |
| **Model 2 (OLS Sweep)** | `var2` | $x_1 - x_3$ | 8 | 165 | None (OLS) | 0.9950 | 0.203 | $0.9927 \pm 0.001$ | $0.2896 \pm 0.04$ | Strong fit |
| **Model 3 (Standard Reg)** | `var1` | $x_1 - x_6$ | 5 | 462 | Lasso ($\alpha \approx 0.0033$) | 0.9774 | 0.247 | $0.9686 \pm 0.006$ | $0.3335 \pm 0.03$ | Optimal sparsity |
| **Model 3 (Standard Reg)** | `var2` | $x_1 - x_3$ | 9 | 220 | Ridge ($\alpha \approx 0.0066$) | 0.9959 | 0.167 | $0.9929 \pm 0.001$ | $0.2835 \pm 0.04$ | Smooth spatial shrinkage |
| **Model 3+ (Tuned Final)** | **`var1`** | **$x_1 - x_6$** | **5** | **462** | **Lasso ($\alpha = 0.00348$)** | **0.9774** | **0.2465** | **0.9689 ± 0.005** | **0.3218 ± 0.03** | **WINNER (Final Submit)** |
| **Model 3+ (Tuned Final)** | **`var2`** | **$x_1 - x_3$** | **9** | **220** | **Ridge ($\alpha = 0.00665$)** | **0.9959** | **0.1673** | **0.9929 ± 0.001** | **0.2835 ± 0.04** | **WINNER (Final Submit)** |
| **Model 4 (Overfit Demo)** | `var1` | $x_1 - x_6$ | 8 | 3,003 | None (OLS) | 0.9998 | 0.002 | $0.6757 \pm 0.089$ | $3.4140 \pm 0.94$ | High variance overfitting |
| **Model 4 (Overfit Demo)** | `var2` | $x_1 - x_3$ | 15 | 816 | None (OLS) | 0.9999 | 0.0001 | $-4.906 \pm 3.82$ | $285.50 \pm 189.2$ | Catastrophic collapse |

---

## 5. Final Model Selection & Justifications

### 5.1 Why Model 3+ is the Definite Winner
Across both tasks, **Model 3+** was selected for final test set inference and submission:

1. **Superior Generalization:** Model 3+ achieves the lowest 5-fold cross-validation Mean Squared Error and highest $R^2$ score across all tested configurations.
2. **Minimal Generalization Gap:**
   - On `var1`: Training $R^2 = 0.9774$ vs CV $R^2 = 0.9689$ ($\Delta = 0.0085$).
   - On `var2`: Training $R^2 = 0.9959$ vs CV $R^2 = 0.9929$ ($\Delta = 0.0030$).
   - These tiny margins prove that the model has not memorized sample quirks and will perform reliably on unseen test data.
3. **Physical Appropriateness:**
   - In `var1`, Lasso selectively retained the true thermodynamic relationships while suppressing cross-term noise.
   - In `var2`, Ridge preserved 3D spatial continuity without allowing high-degree polynomial coefficients to blow up.

### 5.2 Verification of Prediction Files
The final predictions were generated by fitting Model 3+ on 100% of the training data and running inference on the test inputs:
- `BT2024054/BT2024054_pred_var1.csv`
- `BT2024054/BT2024054_pred_var2.csv`

Both files underwent rigorous sanity checks:
- **Format:** Exactly 1 column named `y`, matching the official `sample_submission.csv` specification.
- **Length:** Exactly 1,000 predicted rows matching the test dataset.
- **Null Checks:** Zero NaN, null, or infinite values.
- **Statistical Fidelity:**
  - `var1` test predictions: Mean = $1.28$, Std = $4.21$, Min = $-9.77$, Max = $15.78$ (closely mirroring the training target distribution: Mean = $0.76$, Std = $3.31$).
  - `var2` test predictions: Mean = $1.88$, Std = $6.65$, Min = $-29.53$, Max = $29.26$ (closely mirroring the training target distribution: Mean = $2.40$, Std = $6.41$).

---

## 6. Key Learnings & Takeaways

1. **Data-Driven Discovery Over Static Hints:** Machine learning models must be adapted to empirical data through cross-validation. Initial problem hints can be oversimplified or misleading; trusting cross-validation enabled us to improve $R^2$ from $0.18 \to 0.97$ on `var1` and $0.10 \to 0.99$ on `var2`.
2. **Feature Completeness is Fundamental:** No degree of mathematical tuning can compensate for missing physical variables. Restoring all 6 operational features for the turbine and all 3 spatial coordinates for the geological reservoir was the single most impactful step.
3. **Regularization Geometry Must Match the Problem Domain:**
   - $L_1$ (Lasso) is ideal for high-dimensional combinatorial problems where true relationships are sparse.
   - $L_2$ (Ridge) is ideal for continuous spatial physics where all dimensions interact smoothly.
4. **The Reality of Overfitting:** Model 4 illustrated that achieving $R^2 \approx 1.0$ on training data is not an achievement—it is a hazard. Rigorous cross-validation is the only reliable compass for true model performance.

---

## 7. Submission Checklist & Repository Link

- [x] **Report:** Full technical report covering methodology, experiments, thought process, and results.
- [x] **Prediction Files:** `BT2024054_pred_var1.csv` and `BT2024054_pred_var2.csv` formatted per guidelines.
- [x] **GitHub Repository:** Clean repository containing all four model scripts, training pipeline, and documentation:
  - **URL:** [https://github.com/Ayush-patel9/ML_ASSIGNMENT](https://github.com/Ayush-patel9/ML_ASSIGNMENT)
