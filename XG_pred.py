#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# ---- must match training ----
def add_nan_indicators(X: pd.DataFrame) -> pd.DataFrame:
    nan_ind = X.isna().astype(np.uint8)
    nan_ind.columns = [f"{c}__isnan" for c in nan_ind.columns]
    return pd.concat([X, nan_ind], axis=1)

def load_summary(model_dir: Path) -> dict:
    summ = model_dir / "summary.json"
    if not summ.exists():
        sys.exit(f"Missing summary.json in {model_dir}")
    with open(summ, "r") as f:
        return json.load(f)

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"Features CSV is missing columns: {missing[:8]}{'...' if len(missing)>8 else ''}")
    return df[cols].copy()

def build_inference_matrix(one_row_features: pd.DataFrame,
                           train_df: pd.DataFrame,
                           feat_cols: list[str]) -> np.ndarray:
    """
    Replicate training preprocessing:
      1) select numeric feature columns (exact order from summary.json)
      2) add NaN indicator columns
      3) impute medians computed from TRAINING DATA (for numeric features only)
         - indicator columns are 0/1; they won't be NaN
    Returns a 2D numpy array with 1 row.
    """
    # 1) base features
    X_base = ensure_columns(one_row_features, feat_cols).astype(float)

    # medians from training
    train_medians = train_df[feat_cols].astype(float).median(axis=0, skipna=True)

    # 2) add NaN indicator columns (AFTER selecting base)
    X = add_nan_indicators(X_base)

    # 3) impute medians for base columns; indicators left as-is (0/1)
    for c in feat_cols:
        if pd.isna(X.loc[:, c]).any():
            med = train_medians.get(c, np.nan)
            # If an all-NaN column in training → fallback to 0.0
            if pd.isna(med):
                med = 0.0
            X.loc[:, c] = X.loc[:, c].fillna(float(med))

    # Ensure numeric ndarray in fixed column order: base + flags (that add_nan_indicators created)
    # The indicator order mirrors base order
    indicator_cols = [f"{c}__isnan" for c in feat_cols]
    X = X[feat_cols + indicator_cols]

    return X.values  # shape (1, n_features * 2)

def main():
    ap = argparse.ArgumentParser(description="Predict bottleneck parameters from one-row features using trained XGBoost models.")
    ap.add_argument("--features_csv", required=True, help="One-row CSV produced by extract_features_from_vcf.py")
    ap.add_argument("--train_csv",    required=True, help="Training CSV used for XGB (to compute medians)")
    ap.add_argument("--model_dir",    required=True, help="Directory containing xgb_*.json and summary.json")
    ap.add_argument("--out_prefix",   default="xgb_pred", help="Prefix for outputs (JSON/CSV)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    summary = load_summary(model_dir)

    # Which targets and features were used in training?
    targets = summary.get("targets", ["T_generations", "post_decline_fraction"])
    feat_cols = summary.get("features_used", None)
    if not feat_cols:
        sys.exit("summary.json missing 'features_used'.")

    # Load data
    feat_df = pd.read_csv(args.features_csv)
    train_df = pd.read_csv(args.train_csv)

    # Build inference matrix in the exact same way as training
    X_inf = build_inference_matrix(feat_df, train_df, feat_cols)

    # Load models & predict (one row)
    preds = {}
    for t in targets:
        model_path = model_dir / f"xgb_{t}.json"
        if not model_path.exists():
            sys.exit(f"Missing model file: {model_path}")
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        yhat = float(model.predict(X_inf)[0])
        preds[t] = yhat

    # Save + print
    out_json = f"{args.out_prefix}.json"
    out_csv  = f"{args.out_prefix}.csv"
    with open(out_json, "w") as f:
        json.dump(preds, f, indent=2)
    pd.DataFrame([preds]).to_csv(out_csv, index=False)

    print("=== XGBoost predictions ===")
    for k, v in preds.items():
        print(f"{k}: {v:.6f}")
    print(f"\nSaved: {out_json} and {out_csv}")

if __name__ == "__main__":
    main()
