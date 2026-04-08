"""
Train and evaluate a Random Forest model for aqueous solubility prediction.
Target: logS (log10 of molar solubility in mol/L)

Dataset: AqSolDB (data/AqSolDB_v1.0_min.csv) — split 80/20 into train/test.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.features import featurize_dataframe

DATA_PATH  = "data/AqSolDB_v1.0_min.csv"
MODEL_PATH = "data/model.pkl"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"SMILES": "smiles", "Solubility": "logS"})
    return df[["smiles", "logS"]].dropna()


def train(random_state: int = 42):
    # --- Load & split ---
    df = load_data()
    print(f"Dataset (AqSolDB): {len(df)} molecules")

    print("Featurizing molecules...")
    X, valid_mask = featurize_dataframe(df, smiles_col="smiles")
    y = df["logS"].values[valid_mask]
    print(f"Valid molecules: {valid_mask.sum()} / {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # --- Train ---
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics = {
        "train_rmse": mean_squared_error(y_train, y_pred_train) ** 0.5,
        "test_rmse":  mean_squared_error(y_test,  y_pred_test)  ** 0.5,
        "train_r2":   r2_score(y_train, y_pred_train),
        "test_r2":    r2_score(y_test,  y_pred_test),
        "test_mae":   mean_absolute_error(y_test, y_pred_test),
    }

    print("\n=== Results ===")
    print(f"Train RMSE: {metrics['train_rmse']:.3f}  R²: {metrics['train_r2']:.3f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.3f}  R²: {metrics['test_r2']:.3f}  MAE: {metrics['test_mae']:.3f}")

    # --- Save ---
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"\nModel saved to {MODEL_PATH}")

    _plot_parity(y_test, y_pred_test)
    return model, metrics


def _plot_parity(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", s=20)
    lim = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Measured logS (AqSolDB)")
    ax.set_ylabel("Predicted logS")
    ax.set_title("Solubility Prediction — AqSolDB Test Set Parity Plot")
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"RMSE={rmse:.3f}\nR²={r2:.3f}", transform=ax.transAxes)
    plt.tight_layout()
    out = "data/parity_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Parity plot saved to {out}")
    plt.close()
