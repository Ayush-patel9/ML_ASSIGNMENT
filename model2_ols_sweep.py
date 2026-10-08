import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

ROLL = "BT2024054"
DATA_DIR = "./BT2024054"
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def run(problem, features, degree, tag):
    print(f"\n--- Model 2: {problem.upper()} (deg={degree}, OLS all features) ---")
    train = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_{problem}.csv")
    test = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_{problem}.csv")
    X_tr = train[features].values
    y_tr = train["y"].values
    X_te = test[features].values

    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Xp_tr = pf.fit_transform(X_tr)
    Xp_te = pf.transform(X_te)

    mdl = LinearRegression()
    cv_r2 = cross_val_score(mdl, Xp_tr, y_tr, scoring="r2", cv=kf)
    cv_mse = -cross_val_score(mdl, Xp_tr, y_tr, scoring="neg_mean_squared_error", cv=kf)

    mdl.fit(Xp_tr, y_tr)
    y_hat = mdl.predict(Xp_tr)

    print(f"Features: {len(features)}")
    print(f"Poly terms: {Xp_tr.shape[1]}")
    print(f"CV R2: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"CV MSE: {cv_mse.mean():.4f} +/- {cv_mse.std():.4f}")
    print(f"Train R2: {r2_score(y_tr, y_hat):.4f}")
    print(f"Train MSE: {mean_squared_error(y_tr, y_hat):.4f}")

    preds = mdl.predict(Xp_te)
    out = f"{DATA_DIR}/{ROLL}_pred_{problem}_{tag}.csv"
    pd.DataFrame({"y": preds}).to_csv(out, index=False)
    return cv_r2.mean(), cv_mse.mean()

if __name__ == "__main__":
    run("var1", ["x1", "x2", "x3", "x4", "x5", "x6"], 4, "m2")
    run("var2", ["x1", "x2", "x3"], 8, "m2")
