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
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Empty text"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=2)[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_ids = inputs.word_ids(0)
    entities = []
    current_entity = None
    current_entity_tokens = []
    current_start = None

    for idx, (word_id, pred) in enumerate(zip(word_ids, predictions)):
        if word_id is None:
            continue

        label = label_list[pred]
        token_text = tokens[idx]
        if label.startswith("B-"):
            if current_entity:
                entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
                entities.append({
                    "entity": current_entity,
                    "text": entity_text,
                    "start": current_start,
                    "end": current_start + len(entity_text)
                })
            current_entity = label[2:]
            current_entity_tokens = [token_text]
            entity_text = tokenizer.convert_tokens_to_string([token_text]).strip()
            current_start = text.find(entity_text)
        elif label.startswith("I-") and current_entity:
            current_entity_tokens.append(token_text)
        else:
            if current_entity:
                entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
                entities.append({
                    "entity": current_entity,
                    "text": entity_text,
                    "start": current_start,
                    "end": current_start + len(entity_text)
                })
                current_entity = None
                current_entity_tokens = []
                current_start = None

    if current_entity and current_entity_tokens:
        entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
        entities.append({
            "entity": current_entity,
            "text": entity_text,
            "start": current_start,
            "end": current_start + len(entity_text)
        })

    return jsonify({
        "text": text,
        "entities": entities
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)