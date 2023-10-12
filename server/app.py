# Streamlit app (no longer being used)


# import streamlit as st 
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd
# import torch

# # function to map labels to prediction
# def map_label(prediction):
#     labels = ["toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"] # the labels for the toxic tweets dataset
#     output = []
#     for predict, labels in (zip(prediction, labels)): # zip the prediction and labels together and loop through
#         output.append({'label': labels, 'score': predict})
#     return output

# # sort labels by score in descending order
# def score(item):
#     return item['score']

# # steamlit app that allows users to input text through a text area
# # and select a model from a dropdown menu
# # the app then outputs the labels
# st.title("Sentiment Analysis App")
# text = st.text_area("Input text to get sentiment.", "You are a nice person!")
# model = st.selectbox(
#     'Select the model you want to use below.',
#     ("ac8736/toxic-tweets-fine-tuned-distilbert", 
#      "distilbert-base-uncased-finetuned-sst-2-english", 
#      "cardiffnlp/twitter-roberta-base-sentiment", 
#      "finiteautomata/bertweet-base-sentiment-analysis", "ProsusAI/finbert"))
# st.write('You selected:', model)

# # button to get the sentiment
# if st.button("Get Sentiment"):
#     if model != "ac8736/toxic-tweets-fine-tuned-distilbert": # if the model is not the toxic tweets model
#         # load model using pipeline and get prediction
#         classifier = pipeline(model=model)
#         prediction = classifier(text)[0]["label"]
#         if model == "distilbert-base-uncased-finetuned-sst-2-english": # if statements to maps the prediction to the correct sentiment
#             sentiment = prediction
#             st.write(f"The sentiment is {sentiment}.")
#         elif model == "cardiffnlp/twitter-roberta-base-sentiment":
#             sentiment = "NEGATIVE" if prediction == "LABEL_0" else "POSITIVE" if prediction == "LABEL_2" else "NEUTRAL"
#             st.write(f"The sentiment is {sentiment}.")
#         elif model == "finiteautomata/bertweet-base-sentiment-analysis":
#             sentiment = "NEGATIVE" if prediction == "NEG" else "POSITIVE" if prediction == "POS" else "NEUTRAL"
#             st.write(f"The sentiment is {sentiment}.")
#         elif model == "ProsusAI/finbert":
#             sentiment = prediction.upper()
#             st.write(f"The sentiment is {sentiment}.")
#     else: 
#         # load model using AutoModelForSequenceClassification and get prediction
#         # map the prediction and display the results in a table
#         classifier = AutoModelForSequenceClassification.from_pretrained(model)
#         tokenizer = AutoTokenizer.from_pretrained(model)
#         text_token = tokenizer(text, return_tensors="pt")
#         output = classifier(**text_token)
#         prediction = torch.sigmoid(output.logits)*100 # convert logits to a percentage
#         prediction = prediction.detach().numpy().tolist()[0] # convert prediction to a list
#         labels = map_label(prediction) # map the labels
#         labels.sort(key=score, reverse=True) # sort the labels by score in descending order
        
#         df = pd.DataFrame([(text, labels[0]['label'], f"{round(labels[0]['score'], 3)}%", labels[1]['label'], f"{round(labels[1]['score'], 3)}%")], columns=('tweet/text','label 1', 'score 1', 'label 2', 'score 2'))
#         st.table(df) # display the results in a table
#         st.write("Visit https://huggingface.co/ac8736/toxic-tweets-fine-tuned-distilbert for more information about the model and to view all outputs.")
        
