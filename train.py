from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from evaluate import load

data_path = 'data'
dataset = load_from_disk(data_path)
train_dataset = dataset['train']
validation_dataset = dataset['validation']

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

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

        label_ids += [-100] * (128 - len(label_ids))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset
)

trainer.train()

trainer.save_model("./trained_model")