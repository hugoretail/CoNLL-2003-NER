from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict, Any

class NERModel:
    def __init__(self, model_dir: str, label_list: List[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.label_list = label_list

    def predict(self, text: str) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=2)[0].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(0) if hasattr(inputs, "word_ids") else [None] * len(tokens)
        entities = []
        current_entity = None
        current_entity_tokens = []
        current_start = None

        for idx, (word_id, pred) in enumerate(zip(word_ids, predictions)):
            if word_id is None:
                continue
            label = self.label_list[pred]
            token_text = tokens[idx]
            if label.startswith("B-"):
                if current_entity:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
                    entities.append({
                        "entity": current_entity,
                        "text": entity_text,
                        "start": current_start,
                        "end": current_start + len(entity_text)
                    })
                current_entity = label[2:]
                current_entity_tokens = [token_text]
                entity_text = self.tokenizer.convert_tokens_to_string([token_text]).strip()
                current_start = text.find(entity_text)
            elif label.startswith("I-") and current_entity:
                current_entity_tokens.append(token_text)
            else:
                if current_entity:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
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
            entity_text = self.tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
            entities.append({
                "entity": current_entity,
                "text": entity_text,
                "start": current_start,
                "end": current_start + len(entity_text)
            })
        return entities