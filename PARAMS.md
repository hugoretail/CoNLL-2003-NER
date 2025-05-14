# Training the model attempts with main_old.py

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

# Training the model attempts with main.py

## Attempt 1

### Parameters
```
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,          
    per_device_eval_batch_size=8,
    num_train_epochs=3,                     
    weight_decay=0.001,                      
    gradient_accumulation_steps=2,          
    warmup_steps=500,                    
    logging_steps=200,
    logging_dir="./logs",
    dataloader_pin_memory=False,
    fp16=True            
)
```
### Results

> {'train_runtime': 1663.1504, 'train_samples_per_second': 1.804, 'train_steps_per_second': 0.112, 'train_loss': 0.49085297123078375, 'epoch': 2.96} 100% 186/186 [27:43<00:00,  8.94s/it]

## Attempt 2

### Parameters
```
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,          
    per_device_eval_batch_size=8,
    num_train_epochs=6,                     
    weight_decay=0.001,                      
    gradient_accumulation_steps=4,          
    warmup_steps=500,                    
    logging_steps=200,
    logging_dir="./logs",
    dataloader_pin_memory=False,
    fp16=True            
)
```
### Results

> {'train_runtime': 2672.9288, 'train_samples_per_second': 2.245, 'train_steps_per_second': 0.07, 'train_loss': 0.4548583081973496, 'epoch': 5.83}

> Results evaluation: {'eval_loss': 0.07641442865133286, 'eval_precision': 0.9051339285714286, 'eval_recall': 0.9215909090909091, 'eval_f1': 0.9132882882882882, 'eval_accuracy': 0.9815789473684211, 'eval_runtime': 43.4167, 'eval_samples_per_second': 11.516, 'eval_steps_per_second': 1.451, 'epoch': 5.832}

