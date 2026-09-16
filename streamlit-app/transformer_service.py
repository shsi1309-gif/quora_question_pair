import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = APP_DIR / "transformer_model"
DEFAULT_MODEL_NAME = "cross-encoder/quora-distilroberta-base"
FALLBACK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class TransformerService:
    _instance: Optional["TransformerService"] = None

    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model = None
        self.device = self._detect_device()
        self.model_name = model_name_or_path or self._resolve_model_path()
        self._load_model()

    @classmethod
    def get_instance(cls, model_name_or_path: Optional[str] = None) -> "TransformerService":
        if cls._instance is None:
            cls._instance = cls(model_name_or_path=model_name_or_path)
        return cls._instance

    def _detect_device(self) -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception as e:
            logger.warning(f"Failed to check GPU/MPS devices: {e}")
        return "cpu"

    def _resolve_model_path(self) -> str:
        if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
            logger.info(f"Using fine-tuned local model at {LOCAL_MODEL_DIR}")
            return str(LOCAL_MODEL_DIR)

        configured_model = os.environ.get("TRANSFORMER_MODEL_NAME")
        if configured_model:
            return configured_model

        return DEFAULT_MODEL_NAME

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder model '{self.model_name}' on device '{self.device}'...")
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
            logger.info(f"CrossEncoder model '{self.model_name}' loaded successfully.")
        except Exception as primary_error:
            logger.error(
                f"Failed to load primary model '{self.model_name}': {primary_error}. Trying fallback '{FALLBACK_MODEL_NAME}'..."
            )
            try:
                from sentence_transformers import CrossEncoder

                self.model_name = FALLBACK_MODEL_NAME
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512,
                )
                logger.info(f"Fallback model '{self.model_name}' loaded successfully.")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(
        self, question1: str, question2: str, threshold: float = 0.5
    ) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Transformer model is not loaded.")

        q1 = str(question1 or "").strip()
        q2 = str(question2 or "").strip()

        if not q1 or not q2:
            raise ValueError("Both question1 and question2 must be non-empty strings.")

        import numpy as np

        # CrossEncoder prediction
        raw_score = self.model.predict([(q1, q2)])[0]

        # Convert logits to probability if needed
        # Some models output raw logits, others sigmoid probabilities
        if isinstance(raw_score, (list, np.ndarray)):
            # Multiclass logits or multi-output
            if len(raw_score) == 2:
                # Softmax over [not_duplicate, duplicate]
                exp_scores = np.exp(raw_score - np.max(raw_score))
                probs = exp_scores / np.sum(exp_scores)
                prob_duplicate = float(probs[1])
            else:
                prob_duplicate = float(1.0 / (1.0 + np.exp(-raw_score[0])))
        else:
            score_val = float(raw_score)
            if 0.0 <= score_val <= 1.0:
                prob_duplicate = score_val
            else:
                # Apply sigmoid to raw logit
                prob_duplicate = float(1.0 / (1.0 + np.exp(-score_val)))

        prob_duplicate = round(prob_duplicate, 4)
        is_duplicate = prob_duplicate >= threshold

        # Confidence is the probability of the predicted class
        confidence = prob_duplicate if is_duplicate else round(1.0 - prob_duplicate, 4)

        return {
            "prediction": "duplicate" if is_duplicate else "not_duplicate",
            "confidence": confidence,
            "similarityScore": prob_duplicate,
            "threshold": threshold,
            "modelSource": f"transformer_{Path(self.model_name).name}",
            "device": self.device,
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "modelName": self.model_name,
            "device": self.device,
            "architecture": "CrossEncoder",
        }
