from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

app = Flask(__name__)
CORS(app)

model_path = "../trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

@app.route('/predict', methods=['POST'])
def predict():
  data = request.json
  text = data.get('text', '')
  
  if not text:
    return jsonify({"error": "Empty text"}), 400
  
  tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
  
  with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2)
  