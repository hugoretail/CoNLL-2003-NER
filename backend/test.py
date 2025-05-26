from ner_model import NERModel

LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
MODEL_PATH = "../trained_model"

ner_model = NERModel(MODEL_PATH, LABEL_LIST)

sentences = [
    "Barack Obama was born in Hawaii.",
    "Apple is looking at buying U.K. startup for $1 billion."
]

for sentence in sentences:
    entities = ner_model.predict(sentence)
    print(f"Text: {sentence}")
    print("Entities:", entities)
    print()