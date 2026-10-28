# Malicious URL Detection in IoT

A student mini-project prototype that classifies a URL as **legitimate** or **phishing** using URL-level features and two machine-learning models: Random Forest and XGBoost.

> **Important:** This is an educational prototype, not a security product. A model prediction is not proof that a website is safe. The current model does not fetch or inspect page HTML and does not monitor or block IoT network traffic.

## What is included

- Flask web interface with a manually designed, responsive layout
- URL input and model selection (Random Forest / XGBoost)
- Prediction result and estimated phishing probability
- Dataset training script that saves model artifacts locally
- Input validation and clear limitations on the result page

## Dataset

Place the `Dataset.csv` file you used in Google Colab in the project root. The expected columns include `url`, numeric URL feature columns, and `label` where `0 = legitimate/benign` and `1 = phishing`.

The dataset is intentionally **not included** in this repository. It may be large and should only be redistributed if its license permits it.

## Run locally (Windows)

1. Install Python 3.10 or 3.11.
2. Extract this repository folder and open Command Prompt/PowerShell in it.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Copy `Dataset.csv` into the project root.
6. Train the models and create local model files:

   ```powershell
   python train_model.py
   ```

   Training can take several minutes depending on your computer. Model files are saved in `models/` and are excluded from Git by `.gitignore`.

7. Start the website:

   ```powershell
   python app.py
   ```

8. Open the local address printed in the terminal (normally `http://127.0.0.1:5000`).

## GitHub setup

1. Create a **new empty repository** on GitHub, for example `malicious-url-detection-iot`.
2. Do not upload `Dataset.csv` or generated model files unless you have permission and a reason to publish them.
3. Upload the source files and folders from this project (`app.py`, `train_model.py`, `requirements.txt`, `README.md`, `templates/`, `static/`, `.gitignore`).
4. Add a short repository description such as: `A Flask web app comparing Random Forest and XGBoost for URL phishing classification.`
5. In the README, keep the dataset source and license details accurate. Add the dataset's official source link once you have verified it.

## Current limitations to mention in a review

- The CSV contains pre-calculated features. The feature extraction function in `train_model.py` recreates the numeric features from a new URL, but it must be checked against the dataset's original feature-generation code/documentation before claiming exact feature parity.
- The displayed probability is the model's estimated probability, not a calibrated guarantee.
- The model predicts the two labels present in the dataset. Do not claim that it detects every type of malware or IoT attack.
- The app does not inspect webpage content or block network requests.

## Project workflow

URL input → feature extraction → selected trained model → prediction and estimated phishing probability.

## License

Add a license only after confirming the licensing terms for your own code and the dataset. Dataset licensing may be different from the source-code license.
