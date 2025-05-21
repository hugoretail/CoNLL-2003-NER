from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./trained_model")
model = AutoModelForTokenClassification.from_pretrained("./trained_model")
model.eval()

label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

sentences = [
    "Barack Obama was born in Tamriel.",
    "Apple is looking at buying U.K. startup for $1 billion.",
    "Nujabes and J Dilla are not litteraly the same Hip-Hop artists but they are both genius."
]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [label_list[pred] for pred in predictions[0].numpy()]

    print(f"\nPhrase : {sentence}")
    for token, label in zip(tokens, predicted_labels):
        print(f"{token}: {label}")