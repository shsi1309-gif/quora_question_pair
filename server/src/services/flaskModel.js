const flaskModelUrl = process.env.FLASK_MODEL_URL || "http://127.0.0.1:5002";

export async function predictWithFlaskModel(question1, question2) {
  let response;

  try {
    response = await fetch(`${flaskModelUrl}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question1, question2 }),
    });
  } catch (error) {
    throw new Error(
      `Flask model service is not reachable at ${flaskModelUrl}. Start it with: npm run flask:dev`,
    );
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.message || "Flask model prediction failed.");
  }

  return data;
}
