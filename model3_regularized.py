"""
MODEL 3 — REGULARISED POLYNOMIAL (Lasso + Ridge, data-driven)
===============================================================
BT2024054 | Stage 3 of 4

Config (selected via exhaustive 5-fold CV sweep):
  var1: degree=5, features=x1..x6, Lasso (L1) — alpha=0.0033
  var2: degree=9, features=x1,x2,x3, Ridge (L2) — alpha=0.0066

Why regularisation wins over Model 2 (plain OLS):
  - At deg=5 with 462 poly features, OLS can overfit individual noise
  - Lasso zeros out irrelevant cross-interaction terms automatically
  - Ridge smoothly shrinks all 220 coefficients for stable deg-9 fit
  This gives the best generalisation: var1 R²=0.97, var2 R²=0.99
"""

"""
Polynomial Regression  —  ML Assignment 1  |  BT2024054
=========================================================
OPTIMISED MODEL  (data-driven, not PDF hints)

Methodology:
  1. Exhaustive 5-fold CV sweep across:
       - Feature subsets (x1-x3, x1-x4, x1-x5, x1-x6 for var1;
                          x1, x1-x2, x1-x3 for var2)
       - Degrees 1-10 with OLS, then Ridge/Lasso on top candidates
  2. Best config selected purely by CV R²

Results vs baseline (PDF hints — deg 3 OLS x1-x3 / deg 4 OLS x1):
  ┌─────────┬─────────────────────────────────┬──────────┬──────────┐
  │ Problem │ Config                          │ CV R²    │ CV MSE   │
  ├─────────┼─────────────────────────────────┼──────────┼──────────┤
  │ var1    │ BASELINE: deg=3, OLS, x1-x3     │  0.1838  │  8.9044  │
  │ var1    │ OPTIMAL:  deg=5, Lasso, x1-x6   │  0.9684  │  0.3347  │
  ├─────────┼─────────────────────────────────┼──────────┼──────────┤
  │ var2    │ BASELINE: deg=4, OLS, x1        │  0.1058  │ 36.678   │
  │ var2    │ OPTIMAL:  deg=9, Ridge, x1-x3   │  0.9929  │  0.2837  │
  └─────────┴─────────────────────────────────┴──────────┴──────────┘
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings("ignore")

ROLL     = "BT2024054"
DATA_DIR = f"./BT2024054"
kf       = KFold(n_splits=5, shuffle=True, random_state=42)


def load_data(problem):
    train = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_{problem}.csv")
    test  = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_{problem}.csv")
    return train, test


def fit_and_predict(problem, features, degree, regularizer="lasso"):
    print(f"\n{'='*60}")
    print(f" Problem     : {problem.upper()}")
    print(f" Features    : {features}")
    print(f" Degree      : {degree}")
    print(f" Regularizer : {regularizer.upper()}")
    print(f"{'='*60}")

    train, test = load_data(problem)
    X_train = train[features].values
    y_train = train["y"].values
    X_test  = test[features].values

    # Polynomial expansion (no scaling — unscaled Lasso/Ridge converges
    # fine for our feature range [-1, 1] and outperforms scaled version)
    pf       = PolynomialFeatures(degree=degree, include_bias=True)
    Xp_train = pf.fit_transform(X_train)
    Xp_test  = pf.transform(X_test)
    print(f"  Poly features : {Xp_train.shape[1]}")

    # Regularised model with alpha tuned via inner CV
    if regularizer == "lasso":
        model = LassoCV(
            alphas=np.logspace(-4, 1, 80),
            cv=5, max_iter=15000,
            n_jobs=-1, random_state=42
        )
    else:  # ridge
        model = RidgeCV(
            alphas=np.logspace(-4, 4, 80),
            cv=5
        )

    # CV diagnostics
    cv_r2  = cross_val_score(model, Xp_train, y_train,
                              scoring="r2", cv=kf, n_jobs=-1)
    cv_mse = -cross_val_score(model, Xp_train, y_train,
                               scoring="neg_mean_squared_error",
                               cv=kf, n_jobs=-1)
    print(f"  CV R²  : {cv_r2.mean():.5f}  ±  {cv_r2.std():.5f}")
    print(f"  CV MSE : {cv_mse.mean():.5f}  ±  {cv_mse.std():.5f}")

    # Full training
    model.fit(Xp_train, y_train)
    if hasattr(model, "alpha_"):
        print(f"  Best alpha : {model.alpha_:.6f}")

    y_tr     = model.predict(Xp_train)
    tr_mse   = mean_squared_error(y_train, y_tr)
    tr_r2    = r2_score(y_train, y_tr)
    print(f"  Train R²  : {tr_r2:.5f}")
    print(f"  Train MSE : {tr_mse:.5f}")

    return model.predict(Xp_test), cv_r2.mean(), cv_mse.mean(), tr_r2, tr_mse


def save(y_pred, problem):
    path = f"{DATA_DIR}/{ROLL}_pred_{problem}.csv"
    pd.DataFrame({"y": y_pred}).to_csv(path, index=False)
    print(f"  Saved  →  {path}  ({len(y_pred)} rows)")


# ── VAR1: Net Power Score  ─────────────────────────────────────────
#    Degree=5, all 6 features, Lasso
y_v1, cv_r2_v1, cv_mse_v1, tr_r2_v1, tr_mse_v1 = fit_and_predict(
    "var1", ["x1","x2","x3","x4","x5","x6"], degree=5, regularizer="lasso")
save(y_v1, "var1")

# ── VAR2: Thermal Anomaly Score  ───────────────────────────────────
#    Degree=9, all 3 features, Ridge
y_v2, cv_r2_v2, cv_mse_v2, tr_r2_v2, tr_mse_v2 = fit_and_predict(
    "var2", ["x1","x2","x3"], degree=9, regularizer="ridge")
save(y_v2, "var2")

# ── Summary ────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print(" FINAL SUMMARY")
print("="*60)
summary = pd.DataFrame({
    "Problem"     : ["var1 (baseline)", "var1 (OPTIMAL)",
                     "var2 (baseline)", "var2 (OPTIMAL)"],
    "Config"      : ["deg3 OLS x1-x3", f"deg5 Lasso x1-x6",
                     "deg4 OLS x1",    f"deg9 Ridge x1-x3"],
    "CV R²"       : [0.1838, round(cv_r2_v1,4),
                     0.1058, round(cv_r2_v2,4)],
    "CV MSE"      : [8.9044, round(cv_mse_v1,4),
                     36.678, round(cv_mse_v2,4)],
    "Train R²"    : ["—", round(tr_r2_v1,4),
                     "—", round(tr_r2_v2,4)],
})
print(summary.to_string(index=False))
