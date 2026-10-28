from pathlib import Path
import ipaddress
import joblib
import pandas as pd
from flask import Flask, render_template, request
from train_model import extract_url_features

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
app = Flask(__name__)

def load_artifacts():
    needed = [
        MODEL_DIR / "random_forest.joblib",
        MODEL_DIR / "xgboost.joblib",
        MODEL_DIR / "feature_columns.joblib",
    ]
    missing = [str(path.name) for path in needed if not path.exists()]
    if missing:
        return None, f"Model files are missing: {', '.join(missing)}. Run `python train_model.py` first."
    return {
        "Random Forest": joblib.load(needed[0]),
        "XGBoost": joblib.load(needed[1]),
        "features": joblib.load(needed[2]),
    }, None

ARTIFACTS, LOAD_ERROR = load_artifacts()

def validate_url(value):
    value = (value or "").strip()
    if not value:
        return None, "Enter a URL to analyse."
    if len(value) > 2048:
        return None, "This URL is too long. Please enter a URL under 2,048 characters."
    candidate = value if "://" in value else "https://" + value
    from urllib.parse import urlparse
    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None, "Only http:// and https:// URLs are supported."
    if not parsed.hostname or "." not in parsed.hostname:
        try:
            ipaddress.ip_address(parsed.hostname or "")
        except ValueError:
            return None, "Enter a complete domain or IP address, such as example.com."
    if any(ch.isspace() for ch in value):
        return None, "URLs cannot contain spaces. Check the address and try again."
    return candidate, None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_url = ""
    selected_model = request.form.get("model", "XGBoost")
    if selected_model not in {"XGBoost", "Random Forest"}:
        selected_model = "XGBoost"

    if request.method == "POST":
        submitted_url = request.form.get("url", "")
        normalized_url, error = validate_url(submitted_url)
        if not error and LOAD_ERROR:
            error = LOAD_ERROR
        if not error:
            features = extract_url_features(normalized_url)
            cols = ARTIFACTS["features"]
            row = pd.DataFrame([[features.get(col, float("nan")) for col in cols]], columns=cols)
            model = ARTIFACTS[selected_model]
            pred = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][1])
            result = {
                "url": normalized_url,
                "label": "Phishing" if pred == 1 else "Likely legitimate",
                "kind": "danger" if pred == 1 else "safe",
                "probability": f"{probability:.1%}",
                "model": selected_model,
                "note": (
                    "The model flagged this URL based on its URL-level features. Avoid opening it unless independently verified."
                    if pred == 1 else
                    "The model did not flag this URL as phishing. This is not proof that the website is safe."
                ),
            }
    return render_template(
        "index.html",
        result=result,
        error=error,
        submitted_url=submitted_url,
        selected_model=selected_model,
        load_error=LOAD_ERROR,
    )

@app.route("/health")
def health():
    return {"status": "ok", "models_loaded": ARTIFACTS is not None}, 200

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
