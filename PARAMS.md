# Training the model attempts

## Best solution found

## Attempt 1

### Parameters
```
training_args = TrainingArguments(
  output_dir="./results",
  learning_rate=1e-2,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  weight_decay=0.01
)
```
### Results

> {'eval_loss': 2.1419551372528076, 'eval_model_preparation_time': 0.0026, 'eval_runtime': 615.0829, 'eval_samples_per_second': 5.284, 'eval_steps_per_second': 0.332}

## Attempt 2

### Parameters
```
training_args = TrainingArguments(
  output_dir="./results",
  learning_rate=1e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  weight_decay=0.01
)
```
### Results

> {'eval_loss': 2.198659896850586, 'eval_model_preparation_time': 0.0078, 'eval_runtime': 354.9195, 'eval_samples_per_second': 9.157, 'eval_steps_per_second': 0.575}

## Attempt 3

### Parameters
```
training_args = TrainingArguments(
  output_dir="./results",
  learning_rate=1e-1,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  weight_decay=0.01
)
```
### Results

> {'eval_loss': 2.3625435829162598, 'eval_model_preparation_time': 0.0033, 'eval_runtime': 292.0465, 'eval_samples_per_second': 11.128, 'eval_steps_per_second': 0.699}

## Attempt 4

### Parameters
```
training_args = TrainingArguments(
  output_dir="./results",
  learning_rate=1e-3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  weight_decay=0.01
)
```
### Results

> {'eval_loss': 2.075740337371826, 'eval_model_preparation_time': 0.002, 'eval_runtime': 405.1126, 'eval_samples_per_second': 8.022, 'eval_steps_per_second': 0.504}

## Attempt 5

### Parameters
```
training_args = TrainingArguments(
  output_dir="./results",
  learning_rate=1e-4,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  weight_decay=0.01
)
```
### Results

> {'eval_loss': 2.2905378341674805, 'eval_model_preparation_time': 0.0035, 'eval_runtime': 259.973, 'eval_samples_per_second': 12.501, 'eval_steps_per_second': 0.785}

