#!/usr/bin/env python3
"""
Test script to verify the TransformerService on sample Quora pairs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "streamlit-app"
sys.path.insert(0, str(APP_DIR))

from transformer_service import TransformerService

TEST_CASES = [
    (
        "How can I improve my communication skills?",
        "What are the best ways to become a better communicator?",
        "duplicate",
    ),
    (
        "How do I start learning machine learning?",
        "What is the best way for a beginner to learn ML?",
        "duplicate",
    ),
    (
        "How do I learn Python for data science?",
        "Why is Python slower than C++?",
        "not_duplicate",
    ),
    (
        "Who is president of India?",
        "Who is prime minister of India?",
        "not_duplicate",
    ),
    (
        "Do you live in Australia?",
        "Are you living in Australia?",
        "duplicate",
    ),
    (
        "What is the capital of France?",
        "What is the capital city of France?",
        "duplicate",
    ),
    (
        "What is the capital of France?",
        "What is the capital of Germany?",
        "not_duplicate",
    ),
]


def run_tests():
    print("Initializing TransformerService...")
    service = TransformerService.get_instance()
    info = service.get_info()
    print(f"Loaded: {info}")

    correct = 0
    total = len(TEST_CASES)

    print("\n--- Running Evaluation on Test Cases ---")
    for q1, q2, expected in TEST_CASES:
        res = service.predict(q1, q2)
        pred = res["prediction"]
        conf = res["confidence"]
        sim = res["similarityScore"]
        is_correct = pred == expected
        if is_correct:
            correct += 1
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"\n{status}")
        print(f"  Q1: {q1}")
        print(f"  Q2: {q2}")
        print(f"  Expected: {expected} | Predicted: {pred} (Confidence: {conf:.2%}, SimScore: {sim:.4f})")

    accuracy = correct / total
    print(f"\nTotal: {correct}/{total} passed ({accuracy:.1%} accuracy on test suite)")


if __name__ == "__main__":
    run_tests()
