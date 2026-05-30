const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5001/api";

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.message || "Request failed");
  }

  return data;
}

export async function predictQuestionPair(payload) {
  const response = await fetch(`${API_BASE_URL}/question-pairs/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse(response);
}

export async function fetchHistory() {
  const response = await fetch(`${API_BASE_URL}/question-pairs/history`);
  return parseResponse(response);
}
