"""
MODEL 4 — INTENTIONAL OVERFIT (High degree, zero regularisation)
===================================================================
BT2024054 | Stage 4 of 4

Config (deliberately extreme to show overfitting):
  var1: degree=8, features=x1..x6, plain OLS — 3003 poly features!
  var2: degree=15, features=x1,x2,x3, plain OLS — 816 poly features!

Purpose:
  Demonstrates what happens when degree is too high and there is
  NO regularisation. The model memorises training noise:
    - Train R² ≈ 1.0  (perfect fit on training data)
    - CV R²    << 0   (catastrophic generalization failure)
    - CV MSE explodes compared to Models 1-3

  This is the "cautionary tale" — shows why Lasso/Ridge are essential
  at high polynomial degrees.
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
    print(f" MODEL 4 | {problem.upper()} | deg={degree} | OLS (OVERFIT!)")
    print(f" Features: {features}")
    print(f"{'='*55}")
    train = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_{problem}.csv")
    test  = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_{problem}.csv")
    X_tr  = train[features].values;  y_tr = train["y"].values
    X_te  = test[features].values

    pf    = PolynomialFeatures(degree=degree, include_bias=True)
    Xp_tr = pf.fit_transform(X_tr)
    Xp_te = pf.transform(X_te)
    print(f"  Poly features : {Xp_tr.shape[1]}  (intentionally excessive)")

    mdl = LinearRegression()
    cv_r2  = cross_val_score(mdl, Xp_tr, y_tr, scoring="r2", cv=kf)
    cv_mse = -cross_val_score(mdl, Xp_tr, y_tr,
                               scoring="neg_mean_squared_error", cv=kf)
    print(f"  CV R²  : {cv_r2.mean():.5f}  ±  {cv_r2.std():.5f}  << should be BAD")
    print(f"  CV MSE : {cv_mse.mean():.5f}  ±  {cv_mse.std():.5f}  << should be HIGH")

    mdl.fit(Xp_tr, y_tr)
    y_hat = mdl.predict(Xp_tr)
    print(f"  Train R²  : {r2_score(y_tr, y_hat):.5f}  << memorised training data")
    print(f"  Train MSE : {mean_squared_error(y_tr, y_hat):.10f}  << near zero!")

    preds = mdl.predict(Xp_te)
    out   = f"{DATA_DIR}/{ROLL}_pred_{problem}_{tag}.csv"
    pd.DataFrame({"y": preds}).to_csv(out, index=False)
    print(f"  Saved → {out}")
    return cv_r2.mean(), cv_mse.mean()

r4_v1, m4_v1 = run("var1", ["x1","x2","x3","x4","x5","x6"], degree=8, tag="m4")
r4_v2, m4_v2 = run("var2", ["x1","x2","x3"],                degree=15, tag="m4")

print(f"\n MODEL 4 SUMMARY (OVERFIT — high MSE, negative R²):")
print(f"  var1: CV R²={r4_v1:.4f}  CV MSE={m4_v1:.4f}")
print(f"  var2: CV R²={r4_v2:.4f}  CV MSE={m4_v2:.4f}")
