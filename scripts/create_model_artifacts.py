from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "streamlit-app"


EXTRA_STOP_WORDS = {
    "would",
    "could",
    "should",
    "whats",
    "im",
    "youre",
}

HARD_EXAMPLES = [
    (
        "Who is president of India?",
        "Who is prime minister of India?",
        0,
    ),
    (
        "Who is president of India?",
        "Who is prime minster of India?",
        0,
    ),
    (
        "Who is prime minister of India?",
        "Who is prime minister of Pakistan?",
        0,
    ),
    (
        "Who is president of India?",
        "Who is president of Pakistan?",
        0,
    ),
    (
        "Do you live in Australia?",
        "Do you live in India?",
        0,
    ),
    (
        "Do you live in australlia?",
        "Do you live in india?",
        0,
    ),
    (
        "Do you live in Canada?",
        "Do you live in India?",
        0,
    ),
    (
        "Do you live in Australia?",
        "Are you living in Australia?",
        1,
    ),
    (
        "Do you live in India?",
        "Are you living in India?",
        1,
    ),
    (
        "Who is the president of India?",
        "Who is India's president?",
        1,
    ),
    (
        "Who is the prime minister of India?",
        "Who is India's prime minister?",
        1,
    ),
    (
        "How can I improve my communication skills?",
        "What are the best ways to become a better communicator?",
        1,
    ),
    (
        "How do I start learning machine learning?",
        "What is the best way for a beginner to learn ML?",
        1,
    ),
    (
        "How do I learn Python for data science?",
        "Why is Python slower than C++?",
        0,
    ),
]


def reservoir_add(bucket, item, limit, seen_count, rng):
    if len(bucket) < limit:
        bucket.append(item)
        return

    index = rng.randint(0, seen_count - 1)
    if index < limit:
        bucket[index] = item


def load_balanced_rows(csv_path, per_class, seed):
    rng = random.Random(seed)
    buckets = {0: [], 1: []}
    seen = {0: 0, 1: 0}

    with csv_path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        required_columns = {"question1", "question2", "is_duplicate"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

        for row in reader:
            question1 = (row.get("question1") or "").strip()
            question2 = (row.get("question2") or "").strip()

            if not question1 or not question2:
                continue

            label = int(row["is_duplicate"])
            seen[label] += 1
            reservoir_add(buckets[label], (question1, question2, label), per_class, seen[label], rng)

    rows = buckets[0] + buckets[1]
    rng.shuffle(rows)
    return rows, seen


def fit_vectorizer(rows, max_features):
    questions = []
    for question1, question2, _label in rows:
        questions.extend([question1, question2])

    vectorizer = CountVectorizer(max_features=max_features, lowercase=True)
    vectorizer.fit(questions)
    return vectorizer


def build_feature_matrix(rows, label):
    sys.path.insert(0, str(APP_DIR))
    import helper

    features = []
    labels = []

    for index, (question1, question2, row_label) in enumerate(rows, start=1):
        features.append(helper.query_point_creator(question1, question2)[0])
        labels.append(row_label)

        if index % 1000 == 0:
            print(f"Built {label} features for {index:,} rows")

    return np.array(features), np.array(labels)


def main():
    parser = ArgumentParser(description="Create model.pkl, cv.pkl, and stopwords.pkl from Quora CSV.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the Quora train CSV with question1, question2, and is_duplicate columns.",
    )
    parser.add_argument("--per-class", type=int, default=6000)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    APP_DIR.mkdir(exist_ok=True)
    dataset_rows, seen = load_balanced_rows(args.dataset, args.per_class, args.seed)
    dataset_labels = [label for _question1, _question2, label in dataset_rows]
    train_rows, validation_rows = train_test_split(
        dataset_rows,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=dataset_labels,
    )
    train_rows.extend(HARD_EXAMPLES * 80)

    if not train_rows:
        raise ValueError("No usable rows were loaded from the dataset.")

    print(
        "Loaded "
        f"{len(train_rows):,} training rows and {len(validation_rows):,} validation rows "
        f"from {seen[0]:,} non-duplicate and {seen[1]:,} duplicate rows "
        f"plus {len(HARD_EXAMPLES) * 80:,} hard-example rows."
    )

    stop_words = set(ENGLISH_STOP_WORDS) | EXTRA_STOP_WORDS
    with (APP_DIR / "stopwords.pkl").open("wb") as stopwords_file:
        pickle.dump(stop_words, stopwords_file)

    vectorizer = fit_vectorizer(train_rows, args.max_features)
    with (APP_DIR / "cv.pkl").open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    train_features, train_labels = build_feature_matrix(train_rows, "training")
    validation_features, validation_labels = build_feature_matrix(validation_rows, "validation")

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
        class_weight="balanced",
    )
    model.fit(train_features, train_labels)
    validation_predictions = model.predict(validation_features)
    validation_accuracy = accuracy_score(validation_labels, validation_predictions)

    print("\nValidation report")
    print(f"Accuracy: {validation_accuracy:.4f}")
    report = classification_report(
        validation_labels,
        validation_predictions,
        target_names=["not_duplicate", "duplicate"],
        output_dict=True,
    )
    print(
        classification_report(
            validation_labels,
            validation_predictions,
            target_names=["not_duplicate", "duplicate"],
        )
    )

    metrics = {
        "accuracy": validation_accuracy,
        "perClass": report,
        "trainingRows": len(train_rows),
        "validationRows": len(validation_rows),
        "sampledNonDuplicateRows": seen[0],
        "sampledDuplicateRows": seen[1],
        "hardExampleRows": len(HARD_EXAMPLES) * 80,
        "maxFeatures": args.max_features,
        "model": "ExtraTreesClassifier",
    }
    with (APP_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    with (APP_DIR / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Created model artifacts in {APP_DIR}")
    print(f"Saved metrics to {APP_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
