# Installing required libraries
# Before you start, make sure that you have all of the necessary libraries installed. You can run the following commands to install them:
# !pip install --user torch==2.3.1
# !pip install --user datasets==3.2.0
# !pip install --user trl==0.11
# !pip install --user huggingface_hub==0.28.1
# !pip install --user transformers==4.43.4
# !pip install --user peft==0.14.0
# !pip install --user nltk==3.9.1 rouge_score==0.1.2
# !pip install --user bitsandbytes==0.43.1
# !pip install --user matplotlib==3.10.0

import json
from datasets import load_dataset, DatasetDict
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import TrainingArguments
from trl import RewardTrainer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# Disable warnings for a cleaner notebook or console experience
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def save_to_json(data, file_path):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data successfully saved to {file_path}")


def load_from_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


# Load the Dahoas/synthetic-instruct-gptj-pairwise dataset
dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")
# Display the dataset
print(dataset)

# Data set features
# Prompt: A text prompt that the model should respond to
# Chosen: The preferred response to the prompt
# Rejected: The less preferred response to the prompt
for i in range(10):
    print('prompt')
    print(dataset["train"][i]['prompt'],'\n')

    print('chosen')
    print(dataset[ 'train'][i]['chosen'],'\n')

    print('rejected')
    print(dataset[ 'train'][i]['rejected'],'\n')
    print('---------------------------\n')

#Model and tokenizer setup
# Define the model name or path
model_name_or_path = "gpt2"

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = GPT2ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)

# Add special tokens if necessary
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Define the maximum length
max_length = 1024
chosen_samples=get_res( dataset,'chosen')
rejected_samples=get_res( dataset,'rejected')
print('chosen',chosen_samples[0])
print('rejected',rejected_samples[0])

# Define a function to combine 'prompt' with 'chosen' and 'rejected' responses
def add_combined_columns(example):
    # Combine 'prompt' with 'chosen' response, formatting it with "Human:" and "Assistant:" labels
    example['prompt_chosen'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["chosen"]

    # Combine 'prompt' with 'rejected' response, formatting it with "Human:" and "Assistant:" labels
    example['prompt_rejected'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["rejected"]

    # Return the modified example
    return example

# Apply the function to each example in the 'train' split of the dataset
dataset['train'] = dataset['train'].map(add_combined_columns)
get_max_len= lambda samples: max([len(sample) for sample in samples])
get_max_len

print("rejected samples length",get_max_len(rejected_samples))
print("chosen samples length",get_max_len(chosen_samples))

find_short = lambda dataset, max_length: [
    i for i, (chosen, rejected) in enumerate(zip(dataset['prompt_chosen'], dataset['prompt_rejected']))
    if len(chosen) < max_length or len(rejected) < max_length
]

max_length=1024
subset_indices=find_short (dataset['train'], max_length)
dataset['train'] = dataset['train'].select(subset_indices)
subset_indices[0:10]

# Define a preprocessing function to tokenize the 'prompt_chosen' and 'prompt_rejected' keys
def preprocess_function(examples):
    # Tokenize the 'prompt_chosen' text with truncation and padding to the maximum length
    tokenized_chosen = tokenizer(examples['prompt_chosen'], truncation=True, max_length=max_length, padding="max_length")

    # Tokenize the 'prompt_rejected' text with truncation and padding to the maximum length
    tokenized_rejected = tokenizer(examples['prompt_rejected'], truncation=True, max_length=max_length, padding="max_length")

    # Return the tokenized inputs as a dictionary
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],  # Token IDs for 'chosen' responses
        "attention_mask_chosen": tokenized_chosen["attention_mask"],  # Attention masks for 'chosen' responses
        "input_ids_rejected": tokenized_rejected["input_ids"],  # Token IDs for 'rejected' responses
        "attention_mask_rejected": tokenized_rejected["attention_mask"],  # Attention masks for 'rejected' responses
    }

#You can apply the ```reprocess_function``` to one sample:
example=preprocess_function(dataset['train'][0])
example.keys()

train_str={'chosen': [sample for sample in dataset['train'] ['prompt_chosen']], 'rejected':[sample for sample in dataset['train'] ['prompt_rejected']]}
dataset['train'] = dataset['train'].map(preprocess_function, batched=True, remove_columns=['prompt',"chosen", "rejected",'prompt_chosen', 'prompt_rejected'])
dataset.column_names
split_dataset = dataset['train'].train_test_split(test_size=0.2)

# Create a DatasetDict to hold train and test splits
dataset_dict = DatasetDict({
    'train': split_dataset['train'],
    'test': split_dataset['test'],
})

#LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["attn.c_attn", "attn.c_proj"]  # Target attention layers
)

# Training arguments
# Define the training arguments by using the TrainingArguments class from the transformers library.
# These arguments configure various aspects of the training process:
# •	per_device_train_batch_size=3: Sets the batch size per device (GPU/CPU) to 3
# •	num_train_epochs=3: Specifies the number of training epochs and is set to 3.
# •	gradient_accumulation_steps=8: Accumulates gradients over 8 steps before performing a backward/update pass,
# effectively increasing the batch size
# •	learning_rate=1.41e-5: Sets the learning rate for the optimizer to 1.41e-5
# •	output_dir="./model_output3": Specifies the directory where the model checkpoints and other outputs are saved
# •	logging_steps=10: Logs training progress every 10 steps
# •	evaluation_strategy="steps": Sets the evaluation strategy to evaluate the model at regular steps
# •	eval_steps=500: Evaluates the model every 500 steps
# •	save_steps=500: Saves the model checkpoint every 500 steps
# •	save_total_limit=2: Limits the number of saved checkpoints to 2, deleting older checkpoints to save space
# These arguments configure the training loop, including batch size, learning rate, logging, evaluation, and
# checkpoint-saving strategies.

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=3,  # Set to 3
    num_train_epochs=3,  # Set to 3
    gradient_accumulation_steps=8,
    learning_rate=1.41e-5,
    output_dir="./model_output3",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
)

# RewardTrainer
# The RewardTrainer is a specialized trainer that is designed to train models with a reward signal.
# This is often used in reinforcement learning scenarios where the model learns to optimize for better responses.
# It is initialized with several parameters:
# •	model: The model to be trained
# •	args: The training arguments. Typically, an instance of TrainingArguments
# •	tokenizer: The tokenizer used to process the text inputs
# •	train_dataset: The training data set
# •	eval_dataset: The evaluation data set
# •	peft_config: The configuration for LoRA
# The RewardTrainer orchestrates the training process, handling tasks such as batching, optimization, evaluation,
# and saving model checkpoints. It is particularly useful for training models that need to learn from feedback signals,
# improving their ability to generate high-quality responses.
# Initialize RewardTrainer
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    peft_config=peft_config,
)

#The next step is training, saving, and evaluating a model by using the RewardTrainer.
# The trainer.train() method initiates the training process, where the model learns from the training data set,
# optimizing its parameters to improve performance. After training, the trainer.save_model(output_dir)
# method saves the trained model to the specified output directory, allowing for future use or deployment.
# Finally, the trainer.evaluate() method evaluates the model's performance on the evaluation data set,
# returning metrics that provide insights into how well the model performs.
# These metrics are then printed to give a detailed view of the model's evaluation results.
# output_dir="./model_output3"

# # Train the model
# trainer.train()

# # Save the model
# trainer.save_model(output_dir)

# # Evaluate the model
# metrics = trainer.evaluate()
# print(metrics)

# model.config.save_pretrained("./backup")
#Now, download the pretained model. If you have trained the model yourself, you can skip this step.
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VZcK8FJ-kQ3nEJoxWGNYTQ/RetriverTrainerModel.zip

!unzip -o RetriverTrainerModel.zip -d extracted_model

#Evaluating the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2ForSequenceClassification.from_pretrained("./extracted_model/model_output3", num_labels=1).to(DEVICE)
model
log_file = f"extracted_model/model_output3/checkpoint-2500/trainer_state.json"

# Read the log file
with open(log_file, 'r') as f:
    logs = json.load(f)

# Extract training loss values
steps = []
losses = []
for log in logs["log_history"]:
    if "loss" in log:
        steps.append(log["step"])
        losses.append(log["loss"])

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()

text1=train_str['chosen'][0]
print(text1)

inputs = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move inputs to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
logit_1 = outputs.logits
print("Score :",logit_1 )
text2=train_str['rejected'][0]
print(text2)

inputs = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move inputs to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
logit_2 = outputs.logits
print("Score :",logit_2 )

if logit_1 > logit_2:
    print("--------selected---------")
    print(text1, logit_1.detach().item())
    print("--------rejected---------")
    print(text2, logit_2.detach().item())
else:
    print("selected ")
    print(text2, logit_2.detach().item())
    print("rejected")
    print(text2, logit_2.detach().item())

# Function to make a prediction and get the logits
def predict_and_get_logits(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform the forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits from the outputs
    logits = outputs.logits.squeeze().item()  # Assuming binary classification and batch size of 1

    return logits

# Function to compare two texts
def compare_texts(text1, text2):
    logit1 = predict_and_get_logits(text1)
    logit2 = predict_and_get_logits(text2)

    if logit1 > logit2:
        print("selected---------")
        print(text1, f"score: {logit1}")

        return text1
    else:
        print("selected---------")
        print(text2,  f"score: {logit2}")

        return text2

# Define the number of samples to evaluate
N = 10

# Initialize a counter for correct selections
correct_selections = 0

# Iterate over the first N pairs of chosen and rejected responses
for chosen, rejected in zip(train_str['chosen'][0:N], train_str['rejected'][0:N]):
    # Print the chosen response for reference
    print("Chosen Response:\n", chosen)

    # Use the compare_texts function to determine which response is better
    selected_text = compare_texts(chosen, rejected)

    # Check if the selected text is the chosen response
    if selected_text == chosen:
        correct_selections += 1

# Calculate the accuracy as the ratio of correct selections to the total number of samples
accuracy = correct_selections / N

# Print the accuracy
print("Accuracy:", accuracy)


# Evaluate model's preference accuracy on a different subset of data
# 1.	Define a new variable K to set the number of samples for evaluation from a different subset of the data.
# 2.	Initialize a counter to track the number of correct selections made by the model.
# 3.	Iterate over the K pairs of chosen and rejected responses from a different subset of
# the data set (for example, from the middle of the data set).
# 4.	For each pair, use the compare_texts function to determine which response is better.
# 5.	Count the number of times the model correctly identifies the chosen response.
# 6.	Calculate and print the accuracy of the model's preferences on this different subset.

# Define the number of samples to evaluate from a different subset
K = 50

# Initialize a counter for correct selections
correct_selections = 0

# Determine the starting index for the different subset (e.g., middle of the dataset)
start_index = len(train_str['chosen']) // 2

# Iterate over K pairs of chosen and rejected responses from the different subset
for chosen, rejected in zip(train_str['chosen'][start_index:start_index + K], train_str['rejected'][start_index:start_index + K]):
    # Use the compare_texts function to determine which response is better
    selected_text = compare_texts(chosen, rejected)

    # Check if the selected text is the chosen response
    if selected_text == chosen:
        correct_selections += 1

# Calculate the accuracy as the ratio of correct selections to the total number of samples
accuracy = correct_selections / K

# Print the accuracy
print("Accuracy on different subset:", accuracy)