import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

models_to_test = [
    ("CrossEncoder: quora-distilroberta", CrossEncoder("cross-encoder/quora-distilroberta-base")),
    ("CrossEncoder: ms-marco-MiniLM-L-6-v2", CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")),
    ("SentenceTransformer: all-MiniLM-L6-v2", SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")),
]

pairs = [
    ("How can I improve my communication skills?", "What are the best ways to become a better communicator?"),
    ("How do I start learning machine learning?", "What is the best way for a beginner to learn ML?"),
    ("How do I learn Python for data science?", "Why is Python slower than C++?"),
    ("Who is president of India?", "Who is prime minister of India?"),
    ("Do you live in Australia?", "Are you living in Australia?"),
    ("What is the capital of France?", "What is the capital city of France?"),
    ("What is the capital of France?", "What is the capital of Germany?"),
]

for name, model in models_to_test:
    print(f"\n================ {name} ================")
    for q1, q2 in pairs:
        if isinstance(model, CrossEncoder):
            score = model.predict([(q1, q2)])[0]
            import numpy as np
            prob = 1.0 / (1.0 + np.exp(-score)) if not (0.0 <= score <= 1.0) else score
            print(f"Q: '{q1}' vs '{q2}' => Score/Prob: {prob:.4f}")
        else:
            e1 = model.encode(q1)
            e2 = model.encode(q2)
            sim = float(util.cos_sim(e1, e2)[0][0])
            print(f"Q: '{q1}' vs '{q2}' => Cosine Sim: {sim:.4f}")
