# Quora Question Pairs Duplicate Detector

A full-stack application for checking whether two Quora-style questions are semantically duplicate.

The notebooks are included for NLP exploration and training experiments. The runnable app uses React, Express, MongoDB, and a Flask model service:

- `client/` - React + Vite frontend
- `server/` - Express API with MongoDB persistence
- `streamlit-app/app.py` - Flask model API
- `streamlit-app/helper.py` - feature engineering code used by the model

Dataset: https://www.kaggle.com/c/quora-question-pairs

## Features

- React interface for comparing question pairs
- Express API endpoint that forwards predictions to Flask
- Flask service converted from the Streamlit model app
- MongoDB-backed recent comparison history
- In-memory history fallback when MongoDB is not connected
- Verdict-only UI: same question or not the same question

## Run Locally

Install dependencies:

```bash
npm run install:all
```

Create the backend environment file:

```bash
cp server/.env.example server/.env
```

Start MongoDB locally, or update `MONGODB_URI` in `server/.env` with your MongoDB Atlas connection string.

The model artifacts are stored in `streamlit-app/`:

```text
streamlit-app/model.pkl
streamlit-app/cv.pkl
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

## Note On The Original Python Model

The Flask API loads `model.pkl`, `cv.pkl`, and `stopwords.pkl` from `streamlit-app/`. To regenerate those artifacts from a Quora training CSV:

```bash
.venv/bin/python scripts/create_model_artifacts.py "/path/to/train.csv"
```
