---
title: Sentiment Analysis App
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
license: mit
---

## Google Sites Link

https://sites.google.com/nyu.edu/sentiment-analysis-app/home

## Hugging Space Link

https://huggingface.co/spaces/ac8736/sentiment-analysis-app

## Model and Problem

The problem we are trying to tackle is classification of sentiments on a given text. The goal was to evaluate the toxicity class of a text, and identify it as either toxic, severely toxic, obscene, insult, threat, identity hate. The model DistilBert was fine tuned with a training set from Kaggle's Toxic Tweets competition for multilabel classification on the provided labels.

## Model Accuracy on a Test Set

Model was evaluated on a test set (20% from the original train.csv file) with an accuracy of 93.282%.

```python
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=.2)

predictions = []
for text in test_texts:
  batch = tokenizer(text, truncation=True, padding='max_length', return_tensors="pt").to(device)
  with torch.no_grad():
    outputs = classifier(**batch)
    prediction = torch.sigmoid(outputs.logits)
    prediction = (prediction > 0.5).float()
    prediction = prediction.cpu().detach().numpy().tolist()[0]
    predictions.append(prediction)

print(accuracy_score(test_labels, predictions))
```

## Expected Output

When using a pretrained model from Hugging Face, below are the expected output. Depending on the model, the label value can be different. But generally, the models follow this format using the pipeline API.

```json
{
  "label": "POS",
  "score": "0.8624%"
}
```

When using the fine tuned model, the output is the following. There are 6 items that is returned, each as an object with label and score. Each item represents a label and its corresponding probability score.

```json
[
  {
    "label": "toxic",
    "score": 0.01677067019045353
  },
  {
    "label": "obscene",
    "score": 0.001478900434449315
  },
  {
    "label": "insult",
    "score": 0.0005515297525562346
  },
  {
    "label": "threat",
    "score": 0.0002597073616925627
  },
  {
    "label": "identity hate",
    "score": 0.00010280739661538973
  },
  {
    "label": "severely toxic",
    "score": 0.000017059319361578673
  }
]
```

## Intructions on Installing Docker on Mac

1. Go to the Docker Desktop install page and select the appropriate chip for your Mac device. If you are on Windows, there is another set of instructions you have to follow.

```
https://www.docker.com/products/docker-desktop/
```

2. Open Docker Desktop and go through the tutorial if you wish. Follow instructions to open a Docker container on your machine.
3. Open terminal and type the following to verify you have Docker running.

```
docker version
```

<img width="571" alt="Screen Shot 2023-03-24 at 11 05 15 AM" src="https://user-images.githubusercontent.com/87680132/227563197-b3c0cc7b-8b4f-4ba0-986e-bb5d13ec3c1f.png">
