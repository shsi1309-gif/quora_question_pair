import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  History,
  RefreshCw,
  SearchCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import { fetchHistory, predictQuestionPair } from "./api.js";

const examples = [
  {
    label: "Career advice",
    question1: "How can I improve my communication skills?",
    question2: "What are the best ways to become a better communicator?",
  },
  {
    label: "Learning path",
    question1: "How do I start learning machine learning?",
    question2: "What is the best way for a beginner to learn ML?",
  },
  {
    label: "Different intent",
    question1: "How do I learn Python for data science?",
    question2: "Why is Python slower than C++?",
  },
];

const initialForm = {
  question1: examples[0].question1,
  question2: examples[0].question2,
};

function wordCount(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [error, setError] = useState("");

  const stats = useMemo(
    () => [
      { label: "Question 1 words", value: wordCount(form.question1) },
      { label: "Question 2 words", value: wordCount(form.question2) },
      { label: "Saved results", value: history.length },
    ],
    [form, history.length],
  );

  async function loadHistory() {
    setIsHistoryLoading(true);
    try {
      const pairs = await fetchHistory();
      setHistory(pairs);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsHistoryLoading(false);
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function useExample(example) {
    setForm({
      question1: example.question1,
      question2: example.question2,
    });
    setResult(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!form.question1.trim() || !form.question2.trim()) {
      setError("Please enter both questions before comparing.");
      return;
    }

    setIsLoading(true);
    try {
      const prediction = await predictQuestionPair(form);
      setResult(prediction);
      await loadHistory();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  const isDuplicate = result?.prediction === "duplicate";

  return (
    <main className="app-page">
      <section className="hero-band">
        <div className="hero-copy">
          <span className="eyebrow">
            <Sparkles size={16} />
            MERN NLP Workspace
          </span>
          <h1>Quora duplicate question detector</h1>
          <p>
            Compare two questions, inspect similarity signals, and save recent predictions through a
            Mongo-backed Express API.
          </p>
        </div>
        <div className="hero-stats" aria-label="Current question statistics">
          {stats.map((stat) => (
            <div className="stat-tile" key={stat.label}>
              <strong>{stat.value}</strong>
              <span>{stat.label}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="workspace-grid">
        <form className="question-panel" onSubmit={handleSubmit}>
          <div className="section-heading">
            <div>
              <span className="eyebrow small">Question Pair</span>
              <h2>Run a comparison</h2>
            </div>
            <SearchCheck size={28} />
          </div>

          <div className="example-row">
            {examples.map((example) => (
              <button type="button" key={example.label} onClick={() => useExample(example)}>
                {example.label}
              </button>
            ))}
          </div>

          <label>
            <span>Question 1</span>
            <textarea
              value={form.question1}
              onChange={(event) => updateField("question1", event.target.value)}
              placeholder="Paste the first question"
            />
          </label>

          <label>
            <span>Question 2</span>
            <textarea
              value={form.question2}
              onChange={(event) => updateField("question2", event.target.value)}
              placeholder="Paste the second question"
            />
          </label>

          {error ? <div className="alert">{error}</div> : null}

          <button className="primary-action" type="submit" disabled={isLoading}>
            {isLoading ? <RefreshCw className="spin" size={18} /> : <Activity size={18} />}
            {isLoading ? "Analyzing" : "Analyze Similarity"}
          </button>
        </form>

        <aside className="result-panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow small">Prediction</span>
              <h2>Similarity result</h2>
            </div>
            {result ? (
              isDuplicate ? (
                <CheckCircle2 className="success" size={28} />
              ) : (
                <XCircle className="warning" size={28} />
              )
            ) : (
              <ArrowRight size={28} />
            )}
          </div>

          {result ? (
            <div className={`prediction-card ${isDuplicate ? "duplicate" : "not-duplicate"}`}>
              <span>Final verdict</span>
              <strong>{isDuplicate ? "Same question" : "Not the same question"}</strong>
              <p>
                {isDuplicate
                  ? "The two questions look close enough to share the same answer."
                  : "The two questions appear to have a meaningful difference in intent."}
              </p>
            </div>
          ) : (
          <div className="empty-state">
            <SearchCheck size={36} />
              <p>Submit a pair of questions to see the model verdict.</p>
          </div>
          )}

        </aside>
      </section>

      <section className="history-panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow small">MongoDB History</span>
            <h2>Recent comparisons</h2>
          </div>
          <button className="icon-action" type="button" onClick={loadHistory} title="Refresh history">
            <History size={19} />
          </button>
        </div>

        {isHistoryLoading ? <p className="muted">Loading history...</p> : null}

        <div className="history-list">
          {history.length ? (
            history.map((item) => (
              <article className="history-item" key={item._id}>
                <div>
                  <strong>
                    {item.prediction === "duplicate" ? "Same question" : "Not the same question"}
                  </strong>
                  <span>{item.modelSource === "python_pickle_model" ? "Original model" : "Final verdict"}</span>
                </div>
                <p>{item.question1}</p>
                <p>{item.question2}</p>
              </article>
            ))
          ) : (
            <p className="muted">No comparisons saved yet.</p>
          )}
        </div>
      </section>
    </main>
  );
}

export default App;
