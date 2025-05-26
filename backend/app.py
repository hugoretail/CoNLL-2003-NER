from flask import Flask, request, jsonify
from flask_cors import CORS
from ner_model import NERModel

app = Flask(__name__)
CORS(app)

LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
MODEL_PATH = "../trained_model"

ner_model = NERModel(MODEL_PATH, LABEL_LIST)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Empty text"}), 400
    entities = ner_model.predict(text)
    return jsonify({
        "text": text,
        "entities": entities
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)