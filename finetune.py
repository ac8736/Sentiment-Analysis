import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd

model_name = "distilbert-base-uncased"

def create_dataset():
    df = pd.read_csv("data/imdb.csv")
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    return df

