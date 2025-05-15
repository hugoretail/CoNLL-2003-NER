from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import load_from_disk
from evaluate import load
import numpy as np
from train import tokenize_and_align_labels

model = AutoModelForTokenClassification.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

dataset = load_from_disk('data')
validation_dataset = dataset['validation']

tokenized_validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)

metric = load("seqeval")
label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[label] for label in label_seq if label != -100]
        for label_seq in labels
    ]
    true_predictions = [
        [label_list[pred] for pred, label in zip(pred_seq, label_seq) if label != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_validation_dataset,
    compute_metrics=compute_metrics
)

eval_results = trainer.evaluate()
print("Results:", eval_results)