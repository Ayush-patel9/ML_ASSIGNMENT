import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

ROLL = "BT2024054"
DATA_DIR = "./BT2024054"
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def fit_and_predict(problem, features, degree, regularizer="lasso"):
    print(f"\n--- Model 3: {problem.upper()} (deg={degree}, {regularizer.upper()}) ---")
    train = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_{problem}.csv")
    test = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_{problem}.csv")

    X_train = train[features].values
    y_train = train["y"].values
    X_test = test[features].values

    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Xp_train = pf.fit_transform(X_train)
    Xp_test = pf.transform(X_test)

    if regularizer == "lasso":
        model = LassoCV(alphas=np.logspace(-4, 1, 80), cv=5, max_iter=20000, random_state=42)
    else:
        model = RidgeCV(alphas=np.logspace(-4, 4, 80), cv=5)

    cv_r2 = cross_val_score(model, Xp_train, y_train, scoring="r2", cv=kf)
    cv_mse = -cross_val_score(model, Xp_train, y_train, scoring="neg_mean_squared_error", cv=kf)

    model.fit(Xp_train, y_train)
    y_hat = model.predict(Xp_train)

    print(f"Poly terms: {Xp_train.shape[1]}")
    if hasattr(model, "alpha_"):
        print(f"Best alpha: {model.alpha_:.6f}")
    print(f"CV R2: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"CV MSE: {cv_mse.mean():.4f} +/- {cv_mse.std():.4f}")
    print(f"Train R2: {r2_score(y_train, y_hat):.4f}")
    print(f"Train MSE: {mean_squared_error(y_train, y_hat):.4f}")

    preds = model.predict(Xp_test)
    out = f"{DATA_DIR}/{ROLL}_pred_{problem}.csv"
    pd.DataFrame({"y": preds}).to_csv(out, index=False)
    return preds, cv_r2.mean(), cv_mse.mean()

if __name__ == "__main__":
    fit_and_predict("var1", ["x1", "x2", "x3", "x4", "x5", "x6"], degree=5, regularizer="lasso")
    fit_and_predict("var2", ["x1", "x2", "x3"], degree=9, regularizer="ridge")
