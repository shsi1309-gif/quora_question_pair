#!/usr/bin/env python3
"""
Test the Flask model server endpoints.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "streamlit-app"
sys.path.insert(0, str(APP_DIR))

from app import app

client = app.test_client()

print("\n--- 1. Testing /health ---")
health_res = client.get("/health")
print(f"Status code: {health_res.status_code}")
print(f"Response: {json.dumps(health_res.json, indent=2)}")

print("\n--- 2. Testing /predict with duplicate pair ---")
dup_res = client.post(
    "/predict",
    json={
        "question1": "How can I improve my communication skills?",
        "question2": "What are the best ways to become a better communicator?",
    },
)
print(f"Status code: {dup_res.status_code}")
print(f"Response: {json.dumps(dup_res.json, indent=2)}")

print("\n--- 3. Testing /predict with hard negative pair (President vs Prime Minister) ---")
neg_res = client.post(
    "/predict",
    json={
        "question1": "Who is president of India?",
        "question2": "Who is prime minister of India?",
    },
)
print(f"Status code: {neg_res.status_code}")
print(f"Response: {json.dumps(neg_res.json, indent=2)}")

print("\n--- 4. Testing /predict with pronoun flip guard (He vs She) ---")
pronoun_res = client.post(
    "/predict",
    json={
        "question1": "Where does he live?",
        "question2": "Where does she live?",
    },
)
print(f"Status code: {pronoun_res.status_code}")
print(f"Response: {json.dumps(pronoun_res.json, indent=2)}")
