# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare the dataset
dataset = load_dataset("yelp_polarity")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(45000))
val_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(45000, 50000))
test_dataset = tokenized_datasets["test"]

# Define the initial training arguments (default values)
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define the objective function for Optuna
def objective(trial):
    # Reinitialize the model in each trial
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # Define hyperparameters to be tuned
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    
    # Set training arguments dynamically
    training_args.learning_rate = learning_rate
    training_args.num_train_epochs = num_train_epochs
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate(eval_dataset=val_dataset)
    return eval_result["eval_loss"]

# Perform the hyperparameter search using Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)

# Get the best trial
best_trial = study.best_trial

# Print best hyperparameters
print(f"Best learning rate: {best_trial.params['learning_rate']}")
print(f"Best number of epochs: {best_trial.params['num_train_epochs']}")

# Re-train the model with the best hyperparameters
best_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args.learning_rate = best_trial.params['learning_rate']
training_args.num_train_epochs = best_trial.params['num_train_epochs']

trainer = Trainer(
    model=best_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# Save the fine-tuned model to disk for later use
best_model.save_pretrained("fine-tuned-bert-yelp")
tokenizer.save_pretrained("fine-tuned-bert-yelp")

# Evaluate the final model on the test dataset
test_dataloader = DataLoader(test_dataset, batch_size=16)
true_labels = []
predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(**batch)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = batch['label'].cpu().numpy()
        true_labels.extend(labels)
        predictions.extend(predicted_labels)

accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-score: {f1:.4f}")

# Generate and display a confusion matrix
cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap='Blues')
