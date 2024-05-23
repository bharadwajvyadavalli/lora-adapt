from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load a smaller pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Prepare a smaller dataset
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Training for 1 epoch
    per_device_train_batch_size=32,  # Larger batch size
    learning_rate=5e-5,  # Slightly higher learning rate
    logging_dir='./logs',
    logging_steps=10,
)

# Step 4: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Step 5: Train the model
trainer.train()


# Step 6: Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

##########################################################################################
#  Test the fine-tuned model
##########################################################################################

from transformers import pipeline

# Load the fine-tuned model
model_path = "./fine_tuned_model"  # Adjust the path if necessary
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Example text
review_positive = "The movie was fantastic! The performances were stunning and the storyline was captivating."
review_negative = "The movie was horrible! The performances were boring and the storyline was dragging."

# Get prediction
prediction = classifier(review_positive)

# Output the result
print("Review:", review_positive)
print("Sentiment:", "Positive" if prediction[0]['label'] == 'LABEL_1' else "Negative")

##########################################################################################

# Get prediction
prediction = classifier(review_negative)

# Output the result
print("Review:", review_negative)
print("Sentiment:", "Positive" if prediction[0]['label'] == 'LABEL_1' else "Negative")