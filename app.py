import streamlit as st 
from transformers import pipeline
import pandas as pd

st.title("Sentiment Analysis App")

text = st.text_area("Input text to get sentiment.", "You are a nice person!")

model = st.selectbox(
    'Select the model you want to use below.',
    ("ac8736/toxic-tweets-fine-tuned-distilbert", "distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment", "finiteautomata/bertweet-base-sentiment-analysis", "ProsusAI/finbert"))

classifier = pipeline(model=model)

st.write('You selected:', model)

if st.button("Get Sentiment"):
    if model != "ac8736/toxic-tweets-fine-tuned-distilbert":
        prediction = classifier(text)[0]["label"]
        if model == "distilbert-base-uncased-finetuned-sst-2-english":
            sentiment = prediction
            st.write(f"The sentiment is {sentiment}.")
        elif model == "cardiffnlp/twitter-roberta-base-sentiment":
            sentiment = "NEGATIVE" if prediction == "LABEL_0" else "POSITIVE" if prediction == "LABEL_2" else "NEUTRAL"
            st.write(f"The sentiment is {sentiment}.")
        elif model == "finiteautomata/bertweet-base-sentiment-analysis":
            sentiment = "NEGATIVE" if prediction == "NEG" else "POSITIVE" if prediction == "POS" else "NEUTRAL"
            st.write(f"The sentiment is {sentiment}.")
        elif model == "ProsusAI/finbert":
            sentiment = prediction.upper()
            st.write(f"The sentiment is {sentiment}.")
    else:
        prediction = classifier(text)[0]
        df = pd.DataFrame([text, prediction['label'], f"{round(prediction['score']*100, 3)}%"])#, columns=["Tweet/Text", "Highest Toxicity", "Probability"])
        st.table(df)
        st.write("Visit https://huggingface.co/ac8736/toxic-tweets-fine-tuned-distilbert for more information about the model and to view all outputs.")
        
