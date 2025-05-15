from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from datasets import load_from_disk

data_path = 'data'

dataset = load_from_disk(data_path)

train_set = dataset["train"]

model_name = "google-bert/bert-base-cased"
transformer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

print(model)

