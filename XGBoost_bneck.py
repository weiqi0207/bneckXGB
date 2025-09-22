#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

import xgboost as xgb
from xgboost.callback import EarlyStopping

# --------- Utils ---------
def select_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    """Use all numeric columns except targets as features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in target_cols]
    return feats

def add_nan_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """Append boolean NaN indicator columns for each numeric column."""
    nan_ind = X.isna().astype(np.uint8)
    nan_ind.columns = [f"{c}__isnan" for c in nan_ind.columns]
    return pd.concat([X, nan_ind], axis=1)

def eval_regression(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# --------- Model training (no test) ---------
def train_one_target(X_train, y_train, X_val, y_val, seed: int = 42):
    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        n_jobs=4,
        # <-- set as estimator params (not fit kwargs)
        eval_metric="rmse",
        early_stopping_rounds=100,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,              # keep quiet; no callbacks/ESR here
        # no eval_metric / no early_stopping_rounds / no callbacks in fit()
    )

    y_val_pred = model.predict(X_val)
    val_metrics = eval_regression(y_val, y_val_pred)
    return model, val_metrics



def main():
    ap = argparse.ArgumentParser(description="Benchmark XGBoost for bottleneck parameter regression (train/val only).")
    ap.add_argument("--csv", required=True, help="Path to training CSV produced by your simulation.")
    ap.add_argument("--outdir", default="xgb_out", help="Output directory for models and reports.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", nargs="+",
                    default=["T_generations", "post_decline_fraction"],
                    help="Target columns to regress (train one model per target).")
    ap.add_argument("--val_size", type=float, default=0.15, help="Fraction of data for validation (0,1).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    for t in args.targets:
        if t not in df.columns:
            raise SystemExit(f"Target '{t}' not found in {args.csv}. Columns present: {list(df.columns)}")

    feat_cols = select_feature_columns(df, target_cols=args.targets)
    if not feat_cols:
        raise SystemExit("No numeric feature columns found (after excluding targets).")

    X = df[feat_cols].copy()
    Y = df[args.targets].copy()

    # Preprocessing: add NaN flags, then impute numeric with median
    prep = Pipeline(steps=[
        ("add_nan_flags", FunctionTransformer(add_nan_indicators)),
        ("impute", SimpleImputer(strategy="median")),
    ])

    # Single split: Train / Validation only
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=args.val_size, random_state=args.seed
    )

    # Fit preprocessing on training only, transform both
    X_train_p = prep.fit_transform(X_train)
    X_val_p   = prep.transform(X_val)

    summary = {
        "csv": os.path.abspath(args.csv),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "features_used": feat_cols,
        "seed": args.seed,
        "targets": args.targets,
        "metrics": {},
        "mode": "train_val_only"
    }

    for t in args.targets:
        print(f"[Train] Target: {t}")
        model, val_m = train_one_target(
            X_train_p, Y_train[t].values,
            X_val_p,   Y_val[t].values,
            seed=args.seed
        )

        # Save model (JSON is portable)
        model_path = outdir / f"xgb_{t}.json"
        model.save_model(str(model_path))

        # Save feature importance (post-preprocessing columns)
        fscore = model.get_booster().get_score(importance_type="gain")
        imp_df = pd.DataFrame(
            sorted(fscore.items(), key=lambda kv: kv[1], reverse=True),
            columns=["feature", "gain"]
        )
        imp_df.to_csv(outdir / f"feature_importance_{t}.csv", index=False)

        # Store metrics
        summary["metrics"][t] = {"val": val_m}

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results (Validation only) ===")
    for t in args.targets:
        m = summary["metrics"][t]["val"]
        print(f"Target: {t}")
        print(f"  Val -> RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} | R2={m['r2']:.4f}")
    print(f"\nArtifacts saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()