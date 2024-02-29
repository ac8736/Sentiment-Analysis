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
    } catch (error) {
      console.log(error);
      alert("Model is loading...");
    }
  }

  return (
    <div>
      <h1>Sentiment Analysis</h1>
      <div className="card">
        <textarea onChange={(e) => setText(e.target.value)} value={text} cols={45} rows={10} />
        <button onClick={getSentiment}>Get Sentiment</button>
        <div>
          <h2>Results</h2>
          <div>
            <p>Identity Hate: {sentiment ? `${sentiment?.identity_hate}` : "0"}%</p>
            <p>Insult: {sentiment ? `${sentiment?.insult}` : "0"}%</p>
            <p>Toxic: {sentiment ? `${sentiment?.toxic}` : "0"}%</p>
            <p>Threat: {sentiment ? `${sentiment?.threat}` : "0"}%</p>
          </div>
        </div>
      </div>
    </div>
  );
}
