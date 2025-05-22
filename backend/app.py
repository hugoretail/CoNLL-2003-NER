from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

app = Flask(__name__)
CORS(app)

model_path = "../trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

@app.route('/predict', methods=['POST'])
def predict():
    pass #TODO

if __name__ == '__main__':
    app.run(debug=True, port=8080)