# Installing required libraries
# The following required libraries are not pre-installed in the Skills Network Labs environment. You will need to run the following cell to install them.
# Note: In this lab, you don't have a pinned version to demonstrate the latest functionality, but you can always pin versions in your labs.
# !pip install torch==2.3.1
# !pip install --user trl==0.11.4 # for optimization training
# !pip install peft==0.14.0 # for creating LoRA architecture
# !pip install matplotlib==3.9.0
# !pip install pandas
# !pip install numpy==1.26.0
# !pip install --user datasets==3.2.0

!pip install transformers==4.45.2
#!pip install --user trl==0.11.0
##imports
import multiprocessing
import os
import requests
import tarfile
import pandas as pd
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, GPT2Tokenizer, set_seed, GenerationConfig
from trl import DPOConfig, DPOTrainer

## Create and configure the model and tokenizer
# Load the GPT-2 model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load a reference model
model_ref = AutoModelForCausalLM.from_pretrained("gpt2")

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the pad token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token
# Set the padding side to "right" to fix the overflow issue with FP16 training
tokenizer.padding_side = "right"

# Disable the use of the cache during the model's forward pass
model.config.use_cache = False

model

### Quantized model configuration (Optional)
#!pip install -U bitsandbytes # this package is required for quantization
#**_Note:_**  _You can run the installed package by restarting a Kernel._
'''## Quantized model --only available on GPU
from transformers import BitsAndBytesConfig

# Configure the quantization parameters
quantization_config = BitsAndBytesConfig(
    # Load the model in 4-bit quantized format
    load_in_4bit=True,
    # Enable double quantization for better accuracy
    bnb_4bit_use_double_quant=True,
    # Use non-uniform 4-bit quantization (nf4)
    bnb_4bit_quant_type="nf4",
    # Use bfloat16 as the computation data type during quantization
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load GPT-2 model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=quantization_config)

# Load a reference model with the same quantization configuration
model_ref = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=quantization_config)

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the pad token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token
# Set the padding side to "right" to fix the overflow issue with FP16 training
tokenizer.padding_side = "right"

# Disable the use of the cache during the model's forward pass
model.config.use_cache = False'''


## Preprocess data set
# Load the dataset from the specified location
ds = load_dataset("BarraHome/ultrafeedback_binarized")
ds.keys()
ds["train_prefs"][0].keys()
ds["train_prefs"][0]


# You can reduce the volume of data (due to resource limitations) by selecting the first 5% examples from each split of the dataset
for key in ds:
    #cnt = round(ds[key].__len__()*0.05)
    cnt=50
    ds[key] = ds[key].select(range(cnt))

# Define a function to process the data
def process(row):
    # delete unwanted columns
    del row["prompt_id"]
    del row["messages"]
    del row["score_chosen"]
    del row["score_rejected"]
    # retrieve the actual response text
    row["chosen"] = row["chosen"][-1]["content"]
    row["rejected"] = row["rejected"][-1]["content"]

    return row

# Apply the data processing function to the dataset
ds = ds.map(
    process,
    num_proc=multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

# Split the dataset into training and evaluation sets
train_dataset = ds['train_prefs']
eval_dataset = ds['test_prefs']

#Let's check the data record.
train_dataset[0]

#Next, define LoRAConfig for efficient fine-tuning.
# PEFT (Parameter-Efficient Finetuning) configuration
peft_config = LoraConfig(
        # The rank of the low-rank adaptation weights
        r=4,
        # The target modules to apply the low-rank adaptation to
        target_modules=['c_proj','c_attn'],
        # The task type for the low-rank adaptation
        task_type="CAUSAL_LM",
        # The scaling factor for the low-rank adaptation weights
        lora_alpha=8,
        # The dropout probability for the low-rank adaptation weights
        lora_dropout=0.1,
        # The bias mode for the low-rank adaptation
        bias="none",
)

### DPO configuration

First, define training arguments.
# DPO configuration
from peft import get_peft_model
training_args = DPOConfig(
    # The beta parameter for the DPO loss function
    #beta is the temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5 .
    beta=0.1,
    # The output directory for the training
    output_dir="dpo",
    # The number of training epochs
    num_train_epochs=5,
    # The batch size per device during training
    per_device_train_batch_size=1,
    # The batch size per device during evaluation
    per_device_eval_batch_size=1,
    # Whether to remove unused columns from the dataset
    remove_unused_columns=False,
    # The number of steps between logging training progress
    logging_steps=10,
    # The number of gradient accumulation steps
    gradient_accumulation_steps=1,
    # The learning rate for the optimization
    learning_rate=1e-4,
    # The evaluation strategy (e.g., after each step or epoch)
    evaluation_strategy="epoch",
    # The number of warmup steps for the learning rate scheduler
    warmup_steps=2,
    # Whether to use 16-bit (float16) precision
    fp16=False,
    # The number of steps between saving checkpoints
    save_steps=500,
    # The maximum number of checkpoints to keep
    #save_total_limit=2,
    # The reporting backend to use (set to 'none' to disable, you can also report to wandb or tensorboard)
    report_to='none'
)

### DPO training

#Next step is creating the actual trainer using DPOTrainer class.
tokenizer.pad_token = tokenizer.eos_token

# Create a DPO trainer
# This trainer will handle the fine-tuning of the model using the DPO technique
trainer = DPOTrainer(
        # The model to be fine-tuned
        model=model,
        # The reference model (not used in this case because LoRA has been used)
        ref_model=None,
        # The DPO training configuration
        args=training_args,
        # The beta parameter for the DPO loss function

        # The training dataset
        train_dataset=train_dataset,
        # The evaluation dataset
        eval_dataset=eval_dataset,
        # The tokenizer for the model
        tokenizer=tokenizer,
        # The PEFT (Parallel Efficient Finetuning) configuration
        peft_config=peft_config,
        # The maximum prompt length
        #max_prompt_length=512,
        # The maximum sequence length
        max_length=512,
    )

#### Training model
**Keep in mind that training the model on a CPU can be time-consuming and may cause the kernel to crash due to memory issues. If this happens, you can bypass training by loading the pre-trained model provided in the next section and proceed from there.**
# Start the training process
trainer.train()

!python –v
print(hasattr(model, "generate"))
# Retrieve log_history and save it to a dataframe
log = pd.DataFrame(trainer.state.log_history)
log_t = log[log['loss'].notna()]
log_e = log[log['eval_loss'].notna()]

# Plot train and evaluation losses
plt.plot(log_t["epoch"], log_t["loss"], label = "train_loss")
plt.plot(log_e["epoch"], log_e["eval_loss"], label = "eval_loss")
plt.legend()
plt.show()

# Load the trained DPO model you just trained
dpo_model = AutoModelForCausalLM.from_pretrained('./dpo/checkpoint-250')
#### Loading trained model
# Define the URL and the filename
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YIDeT3qihEpWChdXN_RmTg/DPO-tar.gz'
filename = './DPO.tar'

# Download the file
response = requests.get(url)

# Save the file locally
with open(filename, 'wb') as f:
    f.write(response.content)

# Extract the tar file
if tarfile.is_tarfile(filename):
    with tarfile.open(filename, 'r') as tar:
        tar.extractall()
        print("Files extracted:", tar.getnames())
else:
    print("The adownloaded file is not a tar file.")

# Load the trained DPO model tiy just trained
dpo_model = AutoModelForCausalLM.from_pretrained('./DPO')

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set a seed for reproducibility
set_seed(42)


# Define the generation configuration for the DPO model
# This sets the parameters for text generation
generation_config = GenerationConfig(
        # Use sampling to generate diverse text
        do_sample=True,
        # Top-k sampling parameter
        top_k=1,
        # Temperature parameter to control the randomness of the generated text
        temperature=0.1,
        # Maximum number of new tokens to generate
        max_new_tokens=25,
        # Use the end-of-sequence token as the padding token
        pad_token_id=tokenizer.eos_token_id
    )

# Define the input prompt for text generation
PROMPT = "Is a higher octane gasoline better for your car?"
# Encode the prompt using the tokenizer
inputs = tokenizer(PROMPT, return_tensors='pt')

# Generate text using the DPO model
outputs = dpo_model.generate(**inputs, generation_config=generation_config)
# Decode the generated text and print it
print("DPO response:\t",tokenizer.decode(outputs[0], skip_special_tokens=True))

# Load the pre-trained GPT-2 model
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
# Generate text using the GPT-2 model
outputs = gpt2_model.generate(**inputs, generation_config=generation_config)
# Decode the generated text and print it
print("\nGPT2 response:\t",tokenizer.decode(outputs[0], skip_special_tokens=True))


#Preprocess the argilla/ultrafeedback-binarized-preferences-cleaned Dataset
#This data set comprises user-generated prompts along with corresponding responses categorized as either "chosen"
# or "rejected." It provides a rich source of binary feedback, making it ideal for training models to align with user preferences.
#Load the data set from the argilla/ultrafeedback-binarized-preferences-cleaned
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
dataset['train']
##### Set the variable cnt to 50 and then select the first 50 (cnt) examples to reduce the volume of data for resource limitations.
cnt = 50  # You can adjust this count based on your requirements

# Select the first 5% of examples
dataset['train'] = dataset['train'].select(range(cnt))
##### Create a function named `process` that takes a row of data as input. Within this function, remove unwanted columns such as `source, chosen-rating, chosen-model, rejected-rating, and rejected-model`. Then, use the map function to apply the process function to each row in the training data set.
#TODO

def process(row):
    # Delete unwanted columns
    del row["source"]
    del row["chosen-rating"]
    del row["chosen-model"]
    del row["rejected-rating"]
    del row["rejected-model"]

    # Retrieve the actual response text
    row["chosen"] = row["chosen"][-1]["content"]
    row["rejected"] = row["rejected"][-1]["content"]

    return row

# Apply the data processing function to the dataset
dataset['train'] = dataset['train'].map(
    process,
    num_proc=multiprocessing.cpu_count(),
    load_from_cache_file=False,
)
##### Split the data set into training and evaluation sets:
#Calculate the size for the training set as 80% of the total data. The remaining 20% will be used for evaluation.
train_size = int(0.8 * len(dataset['train']))  # 80% for training
eval_size = len(dataset['train']) - train_size  # Remaining 20% for evaluation

train_dataset = dataset['train'].select(range(train_size))
eval_dataset = dataset['train'].select(range(train_size, train_size + eval_size))

train_dataset
train_dataset[0]

### Exercise 2: Prompt Inferencing and Comparison with GPT-2
PROMPT = input()
##### Initialize the GPT-2 Tokenizer
#TODO
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
##### Create a generation_config object to set the parameters for text generation.
# - do_sample=True    (It enables sampling, which allows for more diverse outputs.)
# - top_k=1 (It specifies the number of highest probability vocabulary tokens to consider during generation.)
# - temperature=0.1 (It controls the randomness of the output.)
# - max_new_tokens=25 (It sets the maximum number of new tokens to generate during inference.)
# - pad_token_id=tokenizer.eos_token_id (It specifies the token to use for padding.)

generation_config = GenerationConfig(
    # Use sampling to generate diverse text
    do_sample=True,
    # Top-k sampling parameter: controls the number of highest probability tokens to consider
    top_k=1,
    # Temperature parameter: controls the randomness of the generated text
    temperature=0.1,
    # Maximum number of new tokens to generate
    max_new_tokens=25,
    # Use the end-of-sequence token as the padding token
    pad_token_id=tokenizer.eos_token_id
)

##### Create a function named `generate_dpo_response` that takes a prompt as input and generates a response using the DPO model (`dpo_model`).
def generate_dpo_response(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate text using the DPO model
    outputs = dpo_model.generate(**inputs, generation_config=generation_config)

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

##### Create another function named `generate_gpt2_response` that takes a prompt as input and generates a response using the GPT-2 model (`gpt2_model`).
def generate_gpt2_response(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate text using the GPT-2 model
    outputs = gpt2_model.generate(**inputs, generation_config=generation_config)

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

##### Call both functions with a prompt and compare the responses.
dpo_response = generate_dpo_response(PROMPT)
gpt2_response = generate_gpt2_response(PROMPT)

# Print the responses
print("DPO response:\t", dpo_response)
print("\nGPT-2 response:\t", gpt2_response)