import streamlit as st 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

def map_label(prediction):
    labels = ["toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"]
    output = []
    for predict, labels in (zip(prediction, labels)):
        output.append({'label': labels, 'score': predict})
    return output

def score(item):
    return item['score']

st.title("Sentiment Analysis App")

text = st.text_area("Input text to get sentiment.", "You are a nice person!")

model = st.selectbox(
    'Select the model you want to use below.',
    ("ac8736/toxic-tweets-fine-tuned-distilbert", "distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment", "finiteautomata/bertweet-base-sentiment-analysis", "ProsusAI/finbert"))

st.write('You selected:', model)

if st.button("Get Sentiment"):
    if model != "ac8736/toxic-tweets-fine-tuned-distilbert":
        classifier = pipeline(model=model)
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
        classifier = AutoModelForSequenceClassification.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        text_token = tokenizer(text, return_tensors="pt")
        output = classifier(**text_token)
        prediction = torch.sigmoid(output.logits)*100
        prediction = prediction.detach().numpy().tolist()[0]
        labels = map_label(prediction)
        labels.sort(key=score, reverse=True)
        df = pd.DataFrame([(text, labels[0]['label'], f"{round(labels[0]['score'], 3)}%", labels[1]['label'], f"{round(labels[1]['score'], 3)}%")], columns=('tweet/text','label 1', 'score 1', 'label 2', 'score 2'))
        st.table(df)
        st.write("Visit https://huggingface.co/ac8736/toxic-tweets-fine-tuned-distilbert for more information about the model and to view all outputs.")
        
