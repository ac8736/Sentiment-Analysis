from flask import Blueprint, jsonify, request

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

bp = Blueprint('analysis', __name__, url_prefix='/analysis')

@bp.route('/', methods=['POST'])
def get_sentiment():
    def map_label(prediction):
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] # the labels for the toxic tweets dataset
        output = []
        for predict, labels in (zip(prediction, labels)): # zip the prediction and labels together and loop through
            output.append({'label': labels, 'score': round(predict, 2)})
        return output
    
    request_data = request.get_json()
    text = request_data['text']

    model = "ac8736/toxic-tweets-fine-tuned-distilbert"
    classifier = AutoModelForSequenceClassification.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    text_token = tokenizer(text, return_tensors="pt")
    output = classifier(**text_token)
    prediction = torch.sigmoid(output.logits) * 100          # convert logits to a percentage
    prediction = prediction.detach().numpy().tolist()[0]    # convert prediction to a list
    labels = map_label(prediction)
    
    result = { "text": text, "labels": {} }
    for entry in labels:
        result["labels"][entry['label']] = entry['score']

    return jsonify(result)