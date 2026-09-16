#!/usr/bin/env python3
"""
Fine-tune a Cross-Encoder Transformer for Quora Duplicate Question Pairs.
Saves the fine-tuned model artifacts to streamlit-app/transformer_model/ for direct inference.
"""

from argparse import ArgumentParser
import csv
import json
import logging
import os
from pathlib import Path
import random
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "streamlit-app" / "transformer_model"


def load_quora_samples(csv_path: Path, max_samples: int = None, seed: int = 42):
    """Load question pairs and labels from a Quora train.csv file."""
    rng = random.Random(seed)
    pairs = []

    logger.info(f"Loading data from {csv_path}...")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"question1", "question2", "is_duplicate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        for row in reader:
            q1 = (row.get("question1") or "").strip()
            q2 = (row.get("question2") or "").strip()
            label_str = row.get("is_duplicate")

            if not q1 or not q2 or label_str is None:
                continue

            try:
                label = int(label_str)
                pairs.append((q1, q2, label))
            except ValueError:
                continue

    logger.info(f"Total valid pairs found: {len(pairs):,}")
    rng.shuffle(pairs)

    if max_samples and max_samples < len(pairs):
        pairs = pairs[:max_samples]
        logger.info(f"Subsampled to {len(pairs):,} pairs")

    return pairs


def train(
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    samples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    all_pairs = load_quora_samples(dataset_path, max_samples=samples, seed=seed)

    # 85% train, 15% validation
    val_size = int(len(all_pairs) * 0.15)
    train_pairs = all_pairs[:-val_size]
    val_pairs = all_pairs[-val_size:]

    logger.info(f"Train samples: {len(train_pairs):,}, Validation samples: {len(val_pairs):,}")

    train_examples = [
        InputExample(texts=[q1, q2], label=float(label))
        for q1, q2, label in train_pairs
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    # Evaluator on validation set
    val_sentences1 = [q1 for q1, q2, _ in val_pairs]
    val_sentences2 = [q2 for q1, q2, _ in val_pairs]
    val_labels = [label for _, _, label in val_pairs]

    evaluator = CEBinaryClassificationEvaluator(
        val_sentences1,
        val_sentences2,
        val_labels,
        name="quora-val",
    )

    logger.info(f"Initializing base model '{model_name}'...")
    model = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=512,
        device=device,
    )

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    logger.info(f"Starting training for {epochs} epochs (warmup steps: {warmup_steps})...")

    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=len(train_dataloader) // 2 if len(train_dataloader) > 2 else 1,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
    )

    logger.info(f"Evaluating best model saved at {output_dir}...")
    best_model = CrossEncoder(str(output_dir), device=device)
    val_predictions = best_model.predict(
        list(zip(val_sentences1, val_sentences2)),
        show_progress_bar=False,
    )

    # Calculate metrics
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    probs = 1.0 / (1.0 + np.exp(-val_predictions)) if val_predictions.ndim == 1 else val_predictions
    pred_labels = (probs >= 0.5).astype(int)

    acc = accuracy_score(val_labels, pred_labels)
    roc_auc = roc_auc_score(val_labels, probs)
    report = classification_report(
        val_labels,
        pred_labels,
        target_names=["not_duplicate", "duplicate"],
        output_dict=True,
    )

    logger.info(f"Validation Accuracy: {acc:.4f} | ROC-AUC: {roc_auc:.4f}")

    metrics = {
        "modelName": model_name,
        "validationAccuracy": round(float(acc), 4),
        "validationRocAuc": round(float(roc_auc), 4),
        "classificationReport": report,
        "trainSamples": len(train_pairs),
        "valSamples": len(val_pairs),
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "device": device,
    }

    metrics_file = output_dir / "training_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved fine-tuned model and metrics to {output_dir}")


def main():
    parser = ArgumentParser(description="Fine-tune Cross-Encoder for Quora Question Pairs")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to Quora train.csv",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cross-encoder/quora-distilroberta-base",
        help="Base HuggingFace model (default: cross-encoder/quora-distilroberta-base)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to save the fine-tuned model",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30000,
        help="Number of samples to use (default: 30000; set 0 for all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    train(
        dataset_path=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        samples=args.samples if args.samples > 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
