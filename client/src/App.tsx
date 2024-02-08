import { useState } from "react";
import "./App.css";

interface Sentiment {
  identity_hate: string;
  insult: string;
  toxic: string;
  threat: string;
}

export default function App() {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState<Sentiment | null>(null);

  async function getSentiment() {
    try {
      const response = await fetch("https://api-inference.huggingface.co/models/ac8736/toxic-tweets-fine-tuned-distilbert", {
        headers: { Authorization: `Bearer ${import.meta.env.VITE_API_KEY}` },
        method: "POST",
        body: JSON.stringify(text),
      });
      const result = await response.json();

      const labels: Sentiment = {
        identity_hate: (Number(result[0][4].score) * 100).toString().substring(0, 5),
        insult: (Number(result[0][2].score) * 100).toString().substring(0, 5),
        toxic: (Number(result[0][0].score) * 100).toString().substring(0, 5),
        threat: (Number(result[0][3].score) * 100).toString().substring(0, 5),
      };
      setSentiment(labels);
    } catch {
      alert("Model is loading...");
    }
  }
  console.log(import.meta.env.VITE_API_KEY);
  return (
    <div>
      <h1>Sentiment Analysis</h1>
      <p>Initial requests may fail due to model loading.</p>
      <div className="card">
        <textarea onChange={(e) => setText(e.target.value)} value={text} cols={45} rows={10} />
        <button onClick={getSentiment}>Get Sentiment</button>
        {sentiment && (
          <div>
            <h2>Results</h2>
            <div>
              <p>Identity Hate: {sentiment.identity_hate}%</p>
              <p>Insult: {sentiment.insult}%</p>
              <p>Toxic: {sentiment.threat}%</p>
              <p>Threat: {sentiment.threat}%</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
