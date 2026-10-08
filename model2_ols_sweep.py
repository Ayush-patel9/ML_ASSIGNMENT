"""
MODEL 2 — DATA-DRIVEN OLS (No regularisation yet)
====================================================
BT2024054 | Stage 2 of 4

Config (found via 5-fold CV sweep across degrees 1-12
        and all feature subsets):
  var1: degree=4, features=x1..x6 (all 6), plain OLS
  var2: degree=8, features=x1,x2,x3 (all 3), plain OLS

Key insight vs Model 1:
  - ALL features matter (not just the first 3 / first 1)
  - Higher degree needed to capture the actual polynomial
  - But NO regularisation means some overfitting risk at high poly dims
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
DATA_DIR = "./BT2024054"
kf       = KFold(n_splits=5, shuffle=True, random_state=42)

def run(problem, features, degree, tag):
    print(f"\n{'='*55}")
    print(f" MODEL 2 | {problem.upper()} | deg={degree} | OLS (no reg)")
    print(f" Features: {features}")
    print(f"{'='*55}")
    train = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_{problem}.csv")
    test  = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_{problem}.csv")
    X_tr  = train[features].values;  y_tr = train["y"].values
    X_te  = test[features].values

    pf   = PolynomialFeatures(degree=degree, include_bias=True)
    Xp_tr = pf.fit_transform(X_tr)
    Xp_te = pf.transform(X_te)

    mdl = LinearRegression()
    cv_r2  = cross_val_score(mdl, Xp_tr, y_tr, scoring="r2", cv=kf)
    cv_mse = -cross_val_score(mdl, Xp_tr, y_tr,
                               scoring="neg_mean_squared_error", cv=kf)
    print(f"  Poly features : {Xp_tr.shape[1]}")
    print(f"  CV R²  : {cv_r2.mean():.5f}  ±  {cv_r2.std():.5f}")
    print(f"  CV MSE : {cv_mse.mean():.5f}  ±  {cv_mse.std():.5f}")

    mdl.fit(Xp_tr, y_tr)
    y_hat = mdl.predict(Xp_tr)
    print(f"  Train R²  : {r2_score(y_tr, y_hat):.5f}")
    print(f"  Train MSE : {mean_squared_error(y_tr, y_hat):.5f}")

    preds = mdl.predict(Xp_te)
    out   = f"{DATA_DIR}/{ROLL}_pred_{problem}_{tag}.csv"
    pd.DataFrame({"y": preds}).to_csv(out, index=False)
    print(f"  Saved → {out}")
    return cv_r2.mean(), cv_mse.mean()

r2_v1, m2_v1 = run("var1", ["x1","x2","x3","x4","x5","x6"], 4, "m2")
r2_v2, m2_v2 = run("var2", ["x1","x2","x3"],                8, "m2")

print(f"\n MODEL 2 SUMMARY:")
print(f"  var1: CV R²={r2_v1:.4f}  CV MSE={m2_v1:.4f}")
print(f"  var2: CV R²={r2_v2:.4f}  CV MSE={m2_v2:.4f}")
