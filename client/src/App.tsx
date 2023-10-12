import { useState } from "react";
import "./App.css";

interface Sentiment {
  labels: {
    identity_hate: string;
    insult: string;
    toxic: string;
    threat: string;
  };
}

export default function App() {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState<Sentiment | null>(null);

  async function getSentiment() {
    const response = await fetch("http://127.0.0.1:5000/analysis/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    setSentiment(data);
  }

  return (
    <div>
      <h1>Sentiment Analysis</h1>
      <div className="card">
        <textarea onChange={(e) => setText(e.target.value)} value={text} cols={45} rows={10} />
        <button onClick={getSentiment}>Get Sentiment</button>
        {sentiment && (
          <div>
            <h2>Results</h2>
            <div>
              <p>Identity Hate: {sentiment.labels.identity_hate}%</p>
              <p>Insult: {sentiment.labels.insult}%</p>
              <p>Toxic: {sentiment.labels.toxic}%</p>
              <p>Threat: {sentiment.labels.threat}%</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
