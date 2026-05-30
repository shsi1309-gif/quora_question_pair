from pathlib import Path
import json
import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS

import helper


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
CV_PATH = APP_DIR / "cv.pkl"
STOPWORDS_PATH = APP_DIR / "stopwords.pkl"
METRICS_PATH = APP_DIR / "metrics.json"

app = Flask(__name__)
CORS(app)

model = None


def missing_artifacts():
    required_files = {
        "model.pkl": MODEL_PATH,
        "cv.pkl": CV_PATH,
        "stopwords.pkl": STOPWORDS_PATH,
    }
    return [name for name, path in required_files.items() if not path.exists()]


def load_model():
    global model

    if model is None:
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)

    return model


@app.get("/health")
def health():
    missing_files = missing_artifacts()
    metrics = None
    if METRICS_PATH.exists():
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

    return jsonify(
        {
            "ok": True,
            "service": "quora-question-pairs-flask-model",
            "modelReady": not missing_files,
            "missingArtifacts": missing_files,
            "metrics": metrics,
        }
    )


@app.post("/predict")
def predict():
    missing_files = missing_artifacts()
    if missing_files:
        return (
            jsonify(
                {
                    "message": "Missing original model artifact(s): "
                    + ", ".join(missing_files)
                    + f". Put them in {APP_DIR}.",
                }
            ),
            503,
        )

    payload = request.get_json(silent=True) or {}
    question1 = str(payload.get("question1", "")).strip()
    question2 = str(payload.get("question2", "")).strip()

    if not question1 or not question2:
        return jsonify({"message": "Both questions are required."}), 400

    query = helper.query_point_creator(question1, question2)
    result = int(load_model().predict(query)[0])

    return jsonify(
        {
            "prediction": "duplicate" if result else "not_duplicate",
            "modelSource": "flask_original_pickle_model",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
