# Quora Question Pairs Duplicate Detector

A full-stack application for checking whether two Quora-style questions are semantically duplicate.

The notebooks are included for NLP exploration and training experiments. The runnable app uses React, Express, and a Flask model service:

- `client/` - React + Vite frontend
- `server/` - Express API with in-memory session history
- `streamlit-app/app.py` - Flask model API
- `streamlit-app/transformer_service.py` - Cross-Encoder Transformer inference engine
- `streamlit-app/helper.py` - feature engineering code used by the model

Dataset: https://www.kaggle.com/c/quora-question-pairs

## Features

- React interface for comparing question pairs
- Express API endpoint that forwards predictions to Flask
- High-accuracy Cross-Encoder Transformer engine with confidence scoring
- Flask microservice with live hardware acceleration
- In-memory recent comparison history
- Verdict-only UI: same question or not the same question

## Run Locally

Install dependencies:

```bash
npm run install:all
```

Create the backend environment file:

```bash
cp server/.env.example server/.env 2>/dev/null || true
```

The model artifacts are stored in `streamlit-app/`:

```text
streamlit-app/model.pkl
streamlit-app/cv.pkl
streamlit-app/tfidf.pkl
streamlit-app/stopwords.pkl
```

Run the full MERN app:

```bash
npm run dev
```

Open the frontend at:

```text
http://localhost:5173
```

The API runs at:

```text
http://localhost:5001/api
```

The Flask model service runs at:

```text
http://127.0.0.1:5002
```

## API

Predict duplicate status:

```http
POST /api/question-pairs/predict
Content-Type: application/json

{
  "question1": "How can I improve my communication skills?",
  "question2": "What are the best ways to become a better communicator?"
}
```

Fetch recent comparisons:

```http
GET /api/question-pairs/history
```

## High-Accuracy Transformer Engine

The Flask API utilizes a state-of-the-art **Cross-Encoder Transformer** (`cross-encoder/quora-distilroberta-base`) powered by `sentence-transformers` and PyTorch with Apple Silicon (`mps`) / GPU acceleration:

- **Deep Token-Level Cross Attention**: Captures contextual nuance, synonyms, and semantic differences far beyond Bag-of-Words and TF-IDF.
- **Accurate Confidence Scoring**: Returns calibrated confidence and probability scores.
- **Graceful Fallback**: Automatically falls back to the legacy ensemble (`model.pkl`) if transformer weights are offline.

### Fine-Tuning a Transformer on Custom Quora Data

To fine-tune a Cross-Encoder directly on a Quora `train.csv` dataset:

```bash
.venv/bin/python scripts/train_transformer.py "/path/to/train.csv" --samples 30000 --epochs 3
```

### Legacy Tabular Model (Random Forest + XGBoost)

The legacy training pipeline uses handcrafted NLP features, CountVectorizer, TF-IDF, and a soft-voting ensemble of Random Forest and XGBoost:

```bash
.venv/bin/python scripts/create_model_artifacts.py "/path/to/train.csv"
```

