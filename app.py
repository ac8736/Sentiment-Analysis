import streamlit as st 
from transformers import pipeline

# classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("Sentiment Analysis App")

POSITIVE = "You are a great person!"
NEGATIVE = "You are a terrible person!"

st.caption(POSITIVE)
if st.button("Get sentiment", key=1):
    # st.write(classifier(POSITIVE)[0]['label'])
    st.write("POSITIVE")
st.caption(NEGATIVE)
if st.button("Get sentiment", key=2):
    st.write("POSITIVE")
    # st.write(classifier(NEGATIVE)[0]['label'])