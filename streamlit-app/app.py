import json
import logging
from pathlib import Path
import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS

import helper
from transformer_service import TransformerService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
CV_PATH = APP_DIR / "cv.pkl"
TFIDF_PATH = APP_DIR / "tfidf.pkl"
STOPWORDS_PATH = APP_DIR / "stopwords.pkl"
METRICS_PATH = APP_DIR / "metrics.json"

TRANSFORMER_METRICS_PATH = APP_DIR / "transformer_metrics.json"

app = Flask(__name__)
CORS(app)

legacy_model = None
transformer_service = None

OPPOSING_PRONOUN_PAIRS = {
    frozenset(("he", "she")),
    frozenset(("him", "her")),
    frozenset(("his", "her")),
    frozenset(("himself", "herself")),
}


def missing_legacy_artifacts():
    required_files = {
        "model.pkl": MODEL_PATH,
        "cv.pkl": CV_PATH,
        "tfidf.pkl": TFIDF_PATH,
        "stopwords.pkl": STOPWORDS_PATH,
    }
    return [name for name, path in required_files.items() if not path.exists()]


def get_transformer_service() -> TransformerService:
    global transformer_service
    if transformer_service is None:
        try:
            transformer_service = TransformerService.get_instance()
        except Exception as e:
            logger.error(f"Failed to initialize TransformerService: {e}")
            transformer_service = None
    return transformer_service


def load_legacy_model():
    global legacy_model
    if legacy_model is None and MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as model_file:
            legacy_model = pickle.load(model_file)
    return legacy_model


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def load_transformer_metrics():
    if not TRANSFORMER_METRICS_PATH.exists():
        return None
    with TRANSFORMER_METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def has_opposing_pronoun_swap(question1, question2):
    q1_tokens = helper.preprocess(question1).split()
    q2_tokens = helper.preprocess(question2).split()

    if len(q1_tokens) != len(q2_tokens) or not q1_tokens:
        return False

    changed_pairs = [
        frozenset((token1, token2))
        for token1, token2 in zip(q1_tokens, q2_tokens)
        if token1 != token2
    ]

    return len(changed_pairs) == 1 and changed_pairs[0] in OPPOSING_PRONOUN_PAIRS


@app.get("/health")
def health():
    ts = get_transformer_service()
    transformer_ready = ts is not None and ts.is_ready()
    missing_legacy = missing_legacy_artifacts()
    legacy_ready = not missing_legacy

    return jsonify(
        {
            "ok": True,
            "service": "quora-question-pairs-model-service",
            "modelReady": transformer_ready or legacy_ready,
            "activeEngine": "transformer" if transformer_ready else ("legacy_ensemble" if legacy_ready else "none"),
            "transformer": ts.get_info() if ts else {"ready": False},
            "metrics": load_transformer_metrics() or load_metrics(),
            "transformerMetrics": load_transformer_metrics(),
            "legacyModelReady": legacy_ready,
            "missingLegacyArtifacts": missing_legacy,
            "legacyMetrics": load_metrics(),
        }
    )


@app.get("/metrics")
def get_metrics():
    transformer_m = load_transformer_metrics()
    legacy_m = load_metrics()
    return jsonify(
        {
            "active": transformer_m or legacy_m,
            "transformer": transformer_m,
            "legacy": legacy_m,
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    question1 = str(payload.get("question1", "")).strip()
    question2 = str(payload.get("question2", "")).strip()
    threshold = float(payload.get("threshold", 0.5))

    if not question1 or not question2:
        return jsonify({"message": "Both questions are required."}), 400

    # Guard for simple pronoun flips (e.g., he vs she)
    if has_opposing_pronoun_swap(question1, question2):
        return jsonify(
            {
                "prediction": "not_duplicate",
                "confidence": 0.99,
                "similarityScore": 0.01,
                "modelSource": "pronoun_entity_guard",
            }
        )

    # 1. Primary engine: Transformer (Cross-Encoder)
    ts = get_transformer_service()
    if ts is not None and ts.is_ready():
        try:
            result = ts.predict(question1, question2, threshold=threshold)
            return jsonify(result)
        except Exception as err:
            logger.error(f"Transformer inference error: {err}. Falling back to legacy model.")

    # 2. Fallback engine: Sklearn / XGBoost Ensemble
    missing = missing_legacy_artifacts()
    if missing:
        return (
            jsonify(
                {
                    "message": "Neither Transformer model nor complete legacy artifacts are available. "
                    f"Missing: {', '.join(missing)}",
                }
            ),
            503,
        )

    try:
        query = helper.query_point_creator(question1, question2)
        model = load_legacy_model()
        pred = int(model.predict(query)[0])
        metrics = load_metrics() or {}

        # If probability estimation is available
        confidence = 0.78
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(query)[0]
            confidence = round(float(probs[pred]), 4)
            similarity_score = round(float(probs[1]), 4)
        else:
            similarity_score = 1.0 if pred else 0.0

        return jsonify(
            {
                "prediction": "duplicate" if pred else "not_duplicate",
                "confidence": confidence,
                "similarityScore": similarity_score,
                "modelSource": metrics.get("model", "legacy_voting_classifier"),
            }
        )
    except Exception as err:
        logger.error(f"Legacy model inference error: {err}")
        return jsonify({"message": f"Prediction failed: {str(err)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
