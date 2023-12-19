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

When using the fine tuned model, the output is the following. There are six items that is returned, each as an object with label and score. Each item represents a label and its corresponding probability score.

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
![image](https://github.com/ac8736/sentiment-analysis/assets/87680132/a14ed088-17fe-4990-930a-14a7a23bc6b9)

![image](https://github.com/ac8736/sentiment-analysis/assets/87680132/8cbd374e-b42d-4326-b244-2c510d2bc068)


