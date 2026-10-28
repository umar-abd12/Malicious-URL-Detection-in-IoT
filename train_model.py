"""Train Random Forest and XGBoost from Dataset.csv and save local artifacts."""
from pathlib import Path
import math
import re
from collections import Counter
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Dataset.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
EXCLUDED = {"label", "url", "dom", "tld"}

def entropy_of_string(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())

def extract_url_features(url: str) -> dict:
    """Recreate the numeric feature names used in this project dataset."""
    url = str(url).strip()
    raw = url if "://" in url else "http://" + url
    parsed = urlparse(raw)
    host = parsed.hostname or ""
    ext = tldextract.extract(host)
    domain = ".".join(part for part in (ext.domain, ext.suffix) if part)
    tld = ext.suffix or ""
    path = parsed.path or ""
    query = parsed.query or ""

    letters = sum(ch.isalpha() for ch in raw)
    digits = sum(ch.isdigit() for ch in raw)
    specials = sum(not ch.isalnum() for ch in raw)
    length = len(raw)
    is_ipv4 = bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

    return {
        "url_len": length,
        "dom_len": len(domain),
        "is_ip": int(is_ipv4),
        "tld_len": len(tld),
        "subdom_cnt": max(
            0,
            len([part for part in host.split(".") if part]) -
            len([part for part in domain.split(".") if part])
        ),
        "letter_cnt": letters,
        "digit_cnt": digits,
        "special_cnt": specials,
        "eq_cnt": raw.count("="),
        "qm_cnt": raw.count("?"),
        "amp_cnt": raw.count("&"),
        "dot_cnt": raw.count("."),
        "dash_cnt": raw.count("-"),
        "under_cnt": raw.count("_"),
        "letter_ratio": letters / length if length else 0.0,
        "digit_ratio": digits / length if length else 0.0,
        "spec_ratio": specials / length if length else 0.0,
        "is_https": int(parsed.scheme.lower() == "https"),
        "slash_cnt": raw.count("/"),
        "entropy": entropy_of_string(raw),
        "path_len": len(path),
        "query_len": len(query),
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset.csv was not found. Put the CSV file in the same folder as train_model.py."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"url", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must include columns: {sorted(required)}")

    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    df = df[df["url"] != ""]
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    feature_cols = [col for col in df.columns if col not in EXCLUDED]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=250, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                objective="binary:logistic", eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])
    }

    MODEL_DIR.mkdir(exist_ok=True)
    scores = {}
    for name, model in models.items():
        print(f"Training {name.replace('_', ' ').title()}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision_phishing": round(float(precision_score(y_test, pred, pos_label=1, zero_division=0)), 4),
            "recall_phishing": round(float(recall_score(y_test, pred, pos_label=1, zero_division=0)), 4),
            "f1_phishing": round(float(f1_score(y_test, pred, pos_label=1, zero_division=0)), 4),
        }
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(scores[name])

    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")
    joblib.dump(scores, MODEL_DIR / "metrics.joblib")
    print(f"\nModels saved in: {MODEL_DIR}")
    print("Feature extractor consistency must still be verified against the dataset's original feature definitions.")

if __name__ == "__main__":
    main()
