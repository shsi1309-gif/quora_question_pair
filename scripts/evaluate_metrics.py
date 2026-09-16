#!/usr/bin/env python3
"""
Evaluate accuracy, precision, recall, f1-score, and confusion matrix
for the Quora Question Pairs duplicate detection models.
Saves results to streamlit-app/transformer_metrics.json.
"""

from argparse import ArgumentParser
import csv
import json
import logging
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "streamlit-app"
sys.path.insert(0, str(APP_DIR))

from transformer_service import TransformerService


def fetch_glue_qqp_validation(num_samples: int = 1000):
    """Fetch official GLUE QQP validation question pairs from Hugging Face."""
    logger.info(f"Fetching {num_samples} validation pairs from standard GLUE QQP benchmark...")
    pairs = []
    limit = 100
    offset = 0

    while len(pairs) < num_samples:
        batch_limit = min(limit, num_samples - len(pairs))
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset=nyu-mll%2Fglue&config=qqp&split=validation&offset={offset}&limit={batch_limit}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
                rows = data.get("rows", [])
                if not rows:
                    break
                for item in rows:
                    row = item.get("row", {})
                    q1 = (row.get("question1") or "").strip()
                    q2 = (row.get("question2") or "").strip()
                    label = row.get("label")
                    if q1 and q2 and label is not None:
                        pairs.append((q1, q2, int(label)))
                offset += len(rows)
        except Exception as e:
            logger.warning(f"Error fetching batch at offset {offset}: {e}")
            break

    logger.info(f"Successfully loaded {len(pairs):,} validation pairs.")
    return pairs


def load_pairs_from_csv(csv_path: Path, max_samples: int = None):
    """Load question pairs from local CSV file."""
    logger.info(f"Loading validation pairs from {csv_path}...")
    pairs = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q1 = (row.get("question1") or "").strip()
            q2 = (row.get("question2") or "").strip()
            label = row.get("is_duplicate") or row.get("label")
            if q1 and q2 and label is not None:
                try:
                    pairs.append((q1, q2, int(label)))
                except ValueError:
                    continue
            if max_samples and len(pairs) >= max_samples:
                break
    logger.info(f"Loaded {len(pairs):,} pairs from CSV.")
    return pairs


def compute_metrics_dict(y_true, y_pred, y_probs, inference_time_sec):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["not_duplicate", "duplicate"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = (
        cm[0][0],
        cm[0][1],
        cm[1][0],
        cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0,
    )

    roc_auc = roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else None

    avg_ms = round((inference_time_sec / len(y_true)) * 1000, 2) if y_true else 0

    return {
        "accuracy": round(float(acc), 4),
        "precision": {
            "macro": round(float(prec_macro), 4),
            "weighted": round(float(prec_weighted), 4),
            "not_duplicate": round(float(report["not_duplicate"]["precision"]), 4),
            "duplicate": round(float(report["duplicate"]["precision"]), 4),
        },
        "recall": {
            "macro": round(float(rec_macro), 4),
            "weighted": round(float(rec_weighted), 4),
            "not_duplicate": round(float(report["not_duplicate"]["recall"]), 4),
            "duplicate": round(float(report["duplicate"]["recall"]), 4),
        },
        "f1Score": {
            "macro": round(float(f1_macro), 4),
            "weighted": round(float(f1_weighted), 4),
            "not_duplicate": round(float(report["not_duplicate"]["f1-score"]), 4),
            "duplicate": round(float(report["duplicate"]["f1-score"]), 4),
        },
        "rocAuc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusionMatrix": {
            "trueNegative": tn,
            "falsePositive": fp,
            "falseNegative": fn,
            "truePositive": tp,
            "raw": cm,
        },
        "support": {
            "total": len(y_true),
            "notDuplicate": int(report["not_duplicate"]["support"]),
            "duplicate": int(report["duplicate"]["support"]),
        },
        "performance": {
            "totalTimeSeconds": round(inference_time_sec, 2),
            "avgLatencyMsPerPair": avg_ms,
        },
    }


def print_metrics_table(metrics: dict, title: str):
    print("\n" + "=" * 65)
    print(f" {title.center(63)} ")
    print("=" * 65)
    print(f" Total Evaluation Samples : {metrics['support']['total']:,}")
    print(f"   • Non-duplicates (0)   : {metrics['support']['notDuplicate']:,}")
    print(f"   • Duplicates (1)       : {metrics['support']['duplicate']:,}")
    print("-" * 65)
    print(f" 🎯 OVERALL ACCURACY      : {metrics['accuracy'] * 100:.2f}%")
    if metrics.get("rocAuc") is not None:
        print(f" 📈 ROC-AUC SCORE         : {metrics['rocAuc']:.4f}")
    print(f" ⚡ AVG INFERENCE LATENCY  : {metrics['performance']['avgLatencyMsPerPair']} ms / pair")
    print("-" * 65)
    print(f" {'CLASS':<18} | {'PRECISION':<12} | {'RECALL':<12} | {'F1-SCORE':<10}")
    print("-" * 65)
    print(
        f" {'Not Duplicate':<18} | "
        f"{metrics['precision']['not_duplicate']*100:>10.2f}% | "
        f"{metrics['recall']['not_duplicate']*100:>10.2f}% | "
        f"{metrics['f1Score']['not_duplicate']*100:>8.2f}%"
    )
    print(
        f" {'Duplicate':<18} | "
        f"{metrics['precision']['duplicate']*100:>10.2f}% | "
        f"{metrics['recall']['duplicate']*100:>10.2f}% | "
        f"{metrics['f1Score']['duplicate']*100:>8.2f}%"
    )
    print("-" * 65)
    print(
        f" {'Macro Average':<18} | "
        f"{metrics['precision']['macro']*100:>10.2f}% | "
        f"{metrics['recall']['macro']*100:>10.2f}% | "
        f"{metrics['f1Score']['macro']*100:>8.2f}%"
    )
    print(
        f" {'Weighted Average':<18} | "
        f"{metrics['precision']['weighted']*100:>10.2f}% | "
        f"{metrics['recall']['weighted']*100:>10.2f}% | "
        f"{metrics['f1Score']['weighted']*100:>8.2f}%"
    )
    print("-" * 65)
    cm = metrics["confusionMatrix"]
    print(" Confusion Matrix:")
    print(f"   True Negative (TN): {cm['trueNegative']:<6} | False Positive (FP): {cm['falsePositive']:<6}")
    print(f"   False Negative (FN): {cm['falseNegative']:<5} | True Positive (TP):  {cm['truePositive']:<6}")
    print("=" * 65 + "\n")


def main():
    parser = ArgumentParser(description="Evaluate Accuracy, Precision, Recall, and F1-Score for Quora detector.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to evaluation CSV file.")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate (default: 500).")
    parser.add_argument("--output", type=Path, default=APP_DIR / "transformer_metrics.json", help="Output JSON file path.")
    args = parser.parse_args()

    if args.dataset and args.dataset.exists():
        pairs = load_pairs_from_csv(args.dataset, max_samples=args.samples)
    else:
        pairs = fetch_glue_qqp_validation(num_samples=args.samples)

    if not pairs:
        print("No evaluation pairs available. Exiting.")
        sys.exit(1)

    y_true = [p[2] for p in pairs]

    # Evaluate Transformer
    logger.info("Evaluating Transformer Cross-Encoder model...")
    service = TransformerService.get_instance()

    t0 = time.perf_counter()
    y_pred = []
    y_probs = []

    for q1, q2, _ in pairs:
        res = service.predict(q1, q2)
        y_pred.append(1 if res["prediction"] == "duplicate" else 0)
        y_probs.append(res["similarityScore"])

    inference_time = time.perf_counter() - t0

    metrics = compute_metrics_dict(y_true, y_pred, y_probs, inference_time)
    metrics["modelName"] = service.model_name
    metrics["architecture"] = "CrossEncoder"
    metrics["device"] = service.device
    metrics["evaluatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print_metrics_table(metrics, f"TRANSFORMER EVALUATION ({service.model_name})")

    # Save to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved complete metrics file to {args.output}")


if __name__ == "__main__":
    main()
