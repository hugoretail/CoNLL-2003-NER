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

    sample = split[3]
    print("  Data sample:")
    for key, value in sample.items():
        print(f"    {key}:")
        if isinstance(value, list):
            print(f"      {value}")
        else:
            print(f"      {value}")

    print()

display_dataset_info(dataset, 'train')
display_dataset_info(dataset, 'test')
display_dataset_info(dataset, 'validation')

unique_labels = set()
for example in dataset["train"]:
    unique_labels.update(example["ner_tags"])

print("Unique labels:", sorted(unique_labels))
print("Number of labels:", len(unique_labels))
