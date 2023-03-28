import streamlit as st 
from transformers import pipeline

classifier = pipeline(task="sentiment-analysis")

st.title("Sentiment Analysis App")

POSITIVE = "You are a great person!"
NEGATIVE = "You are a terrible person!"

st.caption(POSITIVE)
if st.button("Get sentiment", key=1):
    st.write(classifier(POSITIVE)[0]['label'])
st.caption(NEGATIVE)
if st.button("Get sentiment", key=2):
    st.write(classifier(NEGATIVE)[0]['label'])