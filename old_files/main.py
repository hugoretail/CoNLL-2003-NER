from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

dataset=load_from_disk('data')
train_dataset=dataset['train']
validation_dataset=dataset['validation']

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
    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(ner_tags):
                    label_ids.append(ner_tags[word_idx])
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        label_ids += [-100] * (128 - len(label_ids)) #the required padding for max length
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

#for small tests
#small_train_dataset = train_dataset.select(range(1000))
#small_validation_dataset = validation_dataset.select(range(500))

tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)
# tokenized_train_dataset = small_train_dataset.map(tokenize_and_align_labels, batched=True)
# tokenized_validation_dataset = small_validation_dataset.map(tokenize_and_align_labels, batched=True)

sample_example = tokenized_train_dataset[0]
sample_tokens = tokenizer.convert_ids_to_tokens(sample_example["input_ids"])
sample_labels = sample_example["labels"]
print("Test for a signle tokenized one:")
for token, label_id in zip(sample_tokens, sample_labels):
    if label_id != -100:
        print(f"{token:<15} => {label_list[label_id]}")
    else:
        print(f"{token:<15} => IGNORE")
print("\n")

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.001,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    logging_steps=200,
    logging_dir="./logs",
    dataloader_pin_memory=False,
    fp16=True
)

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
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()
print("Res:", eval_results)