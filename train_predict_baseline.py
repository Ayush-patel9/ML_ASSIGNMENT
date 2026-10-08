"""
Polynomial Regression - ML Assignment 1  |  BT2024054
=======================================================
BASELINE MODEL — Following PDF hints literally:
  var1: degree=3, features=[x1, x2, x3], plain OLS
  var2: degree=4, features=[x1],          plain OLS

This is the starting point before data-driven optimisation.
Results:
  var1  CV R2 = 0.1838  |  CV MSE = 8.9044
  var2  CV R2 = 0.1058  |  CV MSE = 36.678
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

def fit_ols_poly(problem, features, degree):
    print(f"\n{'='*55}")
    print(f" [BASELINE] {problem.upper()}  |  deg={degree}  |  OLS")
    print(f" Features: {features}")
    print(f"{'='*55}")
    train, test = load_data(problem)
    X_train = train[features].values
    y_train = train["y"].values
    X_test  = test[features].values

    pf       = PolynomialFeatures(degree=degree, include_bias=True)
    Xp_train = pf.fit_transform(X_train)
    Xp_test  = pf.transform(X_test)

    model = LinearRegression()
    cv_r2  = cross_val_score(model, Xp_train, y_train, scoring="r2", cv=kf)
    cv_mse = -cross_val_score(model, Xp_train, y_train,
                               scoring="neg_mean_squared_error", cv=kf)
    print(f"  CV R2  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"  CV MSE : {cv_mse.mean():.4f} +/- {cv_mse.std():.4f}")

    model.fit(Xp_train, y_train)
    y_tr  = model.predict(Xp_train)
    print(f"  Train R2  : {r2_score(y_train, y_tr):.4f}")
    print(f"  Train MSE : {mean_squared_error(y_train, y_tr):.4f}")

    return model.predict(Xp_test), cv_r2.mean(), cv_mse.mean()

# ── VAR1 baseline ─────────────────────────────────────────────────
y_v1, cv_r2_v1, cv_mse_v1 = fit_ols_poly("var1", ["x1","x2","x3"], degree=3)
pd.DataFrame({"y": y_v1}).to_csv(
    f"{DATA_DIR}/{ROLL}_pred_var1_baseline.csv", index=False)
print(f"  Saved baseline predictions -> {ROLL}_pred_var1_baseline.csv")

# ── VAR2 baseline ─────────────────────────────────────────────────
y_v2, cv_r2_v2, cv_mse_v2 = fit_ols_poly("var2", ["x1"], degree=4)
pd.DataFrame({"y": y_v2}).to_csv(
    f"{DATA_DIR}/{ROLL}_pred_var2_baseline.csv", index=False)
print(f"  Saved baseline predictions -> {ROLL}_pred_var2_baseline.csv")

print("\n\n" + "="*55)
print(" BASELINE SUMMARY")
print("="*55)
print(f"  var1 | deg=3 | OLS | x1,x2,x3     | CV R2={cv_r2_v1:.4f} | CV MSE={cv_mse_v1:.4f}")
print(f"  var2 | deg=4 | OLS | x1 only       | CV R2={cv_r2_v2:.4f} | CV MSE={cv_mse_v2:.4f}")
