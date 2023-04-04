import streamlit as st 
# from transformers import pipeline

# classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("Sentiment Analysis App")

text = st.text_input("Input text to get sentiment.")

model = st.selectbox(
    'Select the model you want to use below.',
    ("distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment", "papluca/xlm-roberta-base-language-detection", "ProsusAI/finbert"))

st.write('You selected:', model)

if st.button("Get Sentiment"):
    st.write("The sentiment is .")
# st.caption(POSITIVE)
# if st.button("Get sentiment", key=1):
#     # st.write(classifier(POSITIVE)[0]['label'])
#     st.write("POSITIVE")
# st.caption(NEGATIVE)
# if st.button("Get sentiment", key=2):
#     st.write("POSITIVE")
#     # st.write(classifier(NEGATIVE)[0]['label'])