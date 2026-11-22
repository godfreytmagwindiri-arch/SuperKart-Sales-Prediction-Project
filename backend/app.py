# ============================================================
# SUPERKART – FLASK BACKEND API
# GOOGLE COLAB / DOCKER DEPLOYMENT
# ============================================================

from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent

# Load fitted model and feature schema (pointing to superkart_model.joblib)
model = joblib.load(BASE_DIR / "superkart_model.joblib")

with open(BASE_DIR / "feature_schema.json", encoding="utf-8") as f:
    schema = json.load(f)

FEATURES = schema["features"]
NUMERIC = schema["numeric_features"]
CATEGORICAL = schema["categorical_features"]

# Validate prediction inputs
def prepare_input(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing predictors: {missing}")

    df = df[FEATURES].copy()

    for col in NUMERIC:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = df[col].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values: {col}")
        df[col] = converted

    for col in CATEGORICAL:
        df[col] = df[col].astype("string").astype(object)
        df[col] = df[col].where(df[col].notna(), np.nan)

    return df

# Home endpoint
@app.get("/")
def home():
    return jsonify({
        "application": "SuperKart Sales Prediction API",
        "status": "running"})

# Health check
@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

# Feature schema
@app.get("/v1/schema")
def show_schema():
    return jsonify(schema)

# Single sales prediction
@app.post("/v1/sales")
def predict_sales():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400

        df = prepare_input(pd.DataFrame([payload]))
        prediction = float(model.predict(df)[0])

        if not np.isfinite(prediction):
            raise ValueError("Non-finite prediction")

        return jsonify({
            "Predicted_Sales": round(prediction, 2)})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500

# Batch CSV predictions
@app.post("/v1/salesbatch")
def predict_batch():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "CSV file required"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Upload a CSV file"}), 400

        df = pd.read_csv(file)
        if df.empty or len(df) > 1000:
            raise ValueError("CSV must have 1–1000 rows")

        prepared = prepare_input(df)
        predictions = model.predict(prepared)

        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite predictions")

        df["Predicted_Sales"] = np.round(predictions, 2)

        return jsonify({
            "total_records": len(df),
            "predictions": json.loads(df.to_json(orient="records"))})

    except (ValueError, pd.errors.ParserError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
