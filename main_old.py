from datasets import load_from_disk
from evaluate import load
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from itertools import product
import numpy as np

data_path = 'data'
dataset = load_from_disk(data_path)

train_dataset = dataset['train']
test_dataset = dataset['test']
validation_dataset = dataset['validation']

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base  -cased", num_labels=9)

label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        label_ids += [-100] * (128 - len(label_ids))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)

param_grid = {
    "learning_rate": [1e-2],
    "num_train_epochs": [15, 20],
    "per_device_train_batch_size": [8],
    "weight_decay": [0.01, 0.1]
}

param_combinations = list(product(
    param_grid["learning_rate"],
    param_grid["num_train_epochs"],
    param_grid["per_device_train_batch_size"],
    param_grid["weight_decay"]
))

metric = load("seqeval")

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

with open("results.txt", "w") as results_file:
    for params in param_combinations:
        learning_rate, num_train_epochs, batch_size, weight_decay = params

        results_file.write(f"Testing parameters: lr={learning_rate}, epochs={num_train_epochs}, batch_size={batch_size}, weight_decay={weight_decay}\n")
        print(f"Testing parameters: lr={learning_rate}, epochs={num_train_epochs}, batch_size={batch_size}, weight_decay={weight_decay}")

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_validation_dataset,
            compute_metrics=compute_metrics
        )

        results = trainer.evaluate()
        results_file.write(f"Results: {results}\n\n")
        print(f"Results: {results}")