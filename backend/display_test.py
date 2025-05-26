from datasets import load_from_disk

data_path = '../data'

dataset = load_from_disk(data_path)

def display_dataset_info(dataset, split_name):
    split = dataset[split_name]
    num_rows = len(split)
    num_columns = len(split.features)
    print(f"{split_name.capitalize()} Set:")
    print(f"  Number of lines: {num_rows}")
    print(f"  Number of columns: {num_columns}")
    print(f"  Data sample: {split[100:110]}")
    print()

display_dataset_info(dataset, 'train')
display_dataset_info(dataset, 'test')
display_dataset_info(dataset, 'validation')

unique_labels = set()
for example in dataset["train"]:
    unique_labels.update(example["ner_tags"])

print("Unique labels :", unique_labels)
print("Nbr of labels :", len(unique_labels))