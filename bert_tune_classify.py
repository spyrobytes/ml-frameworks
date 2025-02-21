# Fine tuning BERT for binary classification task
# Use Hugging Face's Transformers library and datasets library
#
# We following the following workflow:
# 1. Load the dataset and split it into training, validation, testing sets
# 2. Tokenize the data: Use the BertTokenizer to preprocess the text.
# 3. Fine-tune the model: Use the Trainer API to fine-tune BERT for binary classification.
# 4. Evaluate the model: Test the model on the testing set to measure its performance.
# 5. Save the fine-tuned model for future use.
# 6. Deploy and infer: Use the fine-tuned model to make predictions on new data.
from datasets import load_dataset
from transformers import (BertTokenizer, BertForSequenceClassification, 
                          Trainer, TrainingArguments)

# We'll use the IMDb dataset for this example
dataset = load_dataset("imdb")

# Split the dataset into training, validation, and testing sets
train_dataset = dataset["train"]
val_dataset = dataset["test"].shuffle().select(range(1000))  # Use a subset for validation
test_dataset = dataset["test"].shuffle().select(range(1000))  # Use a subset for testing

# BERT requires tokenized input. We’ll use the BertTokenizer from 
# Hugging Face to preprocess the text.
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Fine-tune the model
# Load the pre-trained BERT model with a binary classification head
# Hugging Face adds new classification head on top of pre-trained 
# BERT model for task-specific tuning: 
#  - classifier.weight and
#  - classifier.bias
# These parameters are initially randomly initialized
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
# After training, evaluate the model on the testing set to measure its 
# performance.
results = trainer.evaluate(test_dataset)
print(results)

# Save the model: weights and configuration (for later use)
model.save_pretrained("./fine-tuned-bert-imdb")
# Save the tokenizer
tokenizer.save_pretrained("./fine-tuned-bert-imdb")

# Inference with the fine-tuned model
# Use the fine-tuned model to make predictions on new data.
from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline("text-classification", model="./fine-tuned-bert-imdb")

# Make a prediction
prediction = classifier("This movie was fantastic!")
print(prediction)