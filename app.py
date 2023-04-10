import streamlit as st 
from transformers import pipeline

st.title("Sentiment Analysis App")

text = st.text_input("Input text to get sentiment.", placeholder="You are a nice person!")

model = st.selectbox(
    'Select the model you want to use below.',
    ("distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment", "finiteautomata/bertweet-base-sentiment-analysis", "ProsusAI/finbert"))

classifier = pipeline(task="sentiment-analysis", model=model)

st.write('You selected:', model)

if st.button("Get Sentiment"):
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
