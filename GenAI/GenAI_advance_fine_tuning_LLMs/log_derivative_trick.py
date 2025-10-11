# Installing required libraries
# The following required libraries are not preinstalled in the Skills Network Labs environment. You must run the following cell to install them:
# Note: The version has been pinned to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.
# This might take approximately 1 minute.
# !pip install --user torch==2.3.1 torchtext==0.17.2
# !pip install --user datasets==3.2.0
# !pip install --user trl==0.11
# !pip install --user transformers==4.43.4
# !pip install --user nltk==3.9.1 rouge_score==0.1.2
# !pip install --user matplotlib==3.10.0
# !pip install numpy==1.26.0
#
# %load_ext autoreload
# %autoreload 2

import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import os

import tarfile
import pickle
import json
import matplotlib.pyplot as plt
import torch
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# Disable warnings for a cleaner notebook or console experience
def warn(*args, **kwargs):
    pass
warnings.warn = warn

#Defining helper functions
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

def pad_sequence_to_length(tensor, length, pad_token_id):
    padding_length = length - tensor.size(0)
    if padding_length > 0:
        padding = torch.full((padding_length,), pad_token_id, dtype=torch.long, device=tensor.device)
        return torch.cat((tensor, padding))
    return tensor

def pad_list_to_batch_size(tensors, batch_size, pad_token_id):
    max_length = max(t.size(0) for t in tensors)
    padded_tensors = [pad_sequence_to_length(t, max_length, pad_token_id) for t in tensors]

    # Add additional padding-only tensors if needed
    while len(padded_tensors) < batch_size:
        padded_tensors.append(torch.full((max_length,), pad_token_id, dtype=torch.long, device=tensors[0].device))

    return padded_tensors[:batch_size]

def print_ppo_stats(stats, related_to_objective=False):
    print("PPO Training Statistics\n")

    if related_to_objective:
        print("Objective Statistics:")
        print(f"  KL Divergence (objective/kl): {stats['objective/kl']}")
        print(f"  KL Coefficient (objective/kl_coef): {stats['objective/kl_coef']}")
        print(f"  Entropy (objective/entropy): {stats['objective/entropy']}\n")

        print("PPO Losses (Related to Minimizing Objective Function):")
        print(f"  Policy Loss (ppo/loss/policy): {stats['ppo/loss/policy']}")
        print(f"  Value Loss (ppo/loss/value): {stats['ppo/loss/value']}")
        print(f"  Total Loss (ppo/loss/total): {stats['ppo/loss/total']}\n")

        print("PPO Policy Statistics:")
        print(f"  Policy Entropy (ppo/policy/entropy): {stats['ppo/policy/entropy']}")
        print(f"  Approx KL (ppo/policy/approxkl): {stats['ppo/policy/approxkl']}")
        print(f"  Clip Fraction (ppo/policy/clipfrac): {stats['ppo/policy/clipfrac']}\n")
    else:
        print("Reward and Value Function Estimation:")
        print(f"  Mean Non-Score Reward (ppo/mean_non_score_reward): {stats['ppo/mean_non_score_reward']}")
        print(f"  Mean Scores (ppo/mean_scores): {stats['ppo/mean_scores']}")
        print(f"  Std Scores (ppo/std_scores): {stats['ppo/std_scores']}")
        print(f"  Value Prediction (ppo/val/vpred): {stats['ppo/val/vpred']}")
        print(f"  Value Prediction Error (ppo/val/error): {stats['ppo/val/error']}")
        print(f"  Value Prediction Variance (ppo/val/var): {stats['ppo/val/var']}")
        print(f"  Value Prediction Mean (ppo/val/mean): {stats['ppo/val/mean']}")
        print(f"  Explained Variance (ppo/val/var_explained): {stats['ppo/val/var_explained']}\n")

    print("Token Lengths:")
    print(f"  Queries Length Mean (tokens/queries_len_mean): {stats['tokens/queries_len_mean']}")
    print(f"  Responses Length Mean (tokens/responses_len_mean): {stats['tokens/responses_len_mean']}\n")

    print("Time Statistics:")
    print(f"  Total Time (time/ppo/total): {stats['time/ppo/total']} seconds\n")


# Example usage with the provided stats and the flag
## Initializing the PPO configuration, model, and tokenizer
config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5)

config.model_name
#'lvwerra/gpt2-imdb'

sent_kwargs = {"top_k":None, "function_to_apply": "none", "batch_size": 2}

model_1 = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token


# first model
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

#Dataset and dataset tokenization
dataset_name = "imdb"
ds = load_dataset(dataset_name, split = "train")

N = 5
for sample in range(N):
    print('text',ds[sample]['text'])
    print('label',ds[sample]['label'])

# Rename the column "text" to "review"
ds = ds.rename_columns({"text": "review"})
ds

#The dataset is filtered to include only reviews that are longer than 200 characters.
ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

input_min_text_length, input_max_text_length = 2, 8
#Create a ```LengthSampler``` object
input_size = LengthSampler(input_min_text_length, input_max_text_length)
input_size

for i in range(10):
    size=input_size()
    print(f"sample {i} has length {size}\n")

sample=ds[0]
sample

sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
sample["input_ids"]

sample["query"] = tokenizer.decode(sample["input_ids"])
sample["query"]

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

ds = ds.map(tokenize, batched=False)
ds.set_format(type="torch")

ds[0]

for i, sample in enumerate(ds):
    if i >= 5:
        break
    print(f"Sample {i+1}:")
    print(f"Review: {sample['review']}")
    print(f"Input IDs: {sample['input_ids']}")
    print(f"Query: {sample['query']}")
    print("-" * 50)


del(ds)
dataset_name="imdb"
ds = load_dataset(dataset_name, split="train")
ds = ds.rename_columns({"text": "review"})

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8,tokenizer=tokenizer):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds



#Create the dataset object
dataset = build_dataset(config)

#You can see each sample has ```input_ids``` and  ```query```
dataset[0]

## Collator function
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

data = [
    {'input_ids': [1, 2, 3, 4], 'query': "sample text", 'review': "This is a sample review."},
    {'input_ids': [5, 6, 7, 8], 'query': "another sample", 'review': "Another sample review."}
]
#Apply the collator function to the above data
batch = collator(data)
batch


# Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that is particularly well-suited for
# training generative models, including those used for chatbots. It helps address specific challenges in training
# these models, such as maintaining coherent and contextually appropriate dialogues.

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
print("ppo_trainer object ",ppo_trainer)

# Please ignore above warnings as the trl version you installed supports this module.
# Determine the appropriate device (CPU or GPU) for training with the PPO Trainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
     device = 0 if torch.cuda.is_available() else "cpu"
print(device)
# cpu


# ## Reward function
#
# In reinforcement learning with PPO (Proximal Policy Optimization), a reward function is used to provide feedback
# on the quality of the actions taken by the policy. For a generative model like a chatbot, the reward function can
# evaluate the quality of the generated responses. Here’s how the sentiment analysis pipeline can be used as a
# reward function:
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

#You'll get the sentiment value as negative here.
text = "this movie was really bad!!"
sentiment_pipe(text, **sent_kwargs)

text = "this movie was really good!!"
sentiment_pipe(text, **sent_kwargs)

## Generating responses using PPO
### Tokenizing and preparing the input batch
batch = next(iter(ppo_trainer.dataloader))
batch.keys()

# Let's take the first two  sample in the batch
batch = {key: batch[key][0:2] for key in batch}
batch

response_tensors = []
query_tensors =  batch["input_ids"]
query_tensors

get_text = lambda response:''.join([tokenizer.decode(r.squeeze()) for r in response])
get_text(query_tensors)

# The dictionary `generation_kwargs` sets the parameters for generating a sequence from the LLM (Language Model). The parameters include:
# - `"min_length": -1` - No minimum length for the generated text.
# - `"top_k": 0.0` - No filtering of the top-k most probable tokens.
# - `"top_p": 1.0` - No nucleus sampling, using the entire distribution.
# - `"do_sample": True` - Enables sampling, allowing for varied responses.
# - `"pad_token_id": 50256` - ID of the padding token, ensuring uniform length across sequences.

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 50256,
}
generation_kwargs

output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

gen_len = output_length_sampler()
gen_len
#9

generation_kwargs["max_new_tokens"] = gen_len
generation_kwargs

query=query_tensors[0]
query

response = ppo_trainer.generate(query, **generation_kwargs)
response


print("query:",get_text(query))
print("response:", get_text(response))

response_tensors.append(response.squeeze()[-gen_len:])
print("newly generated tokens form response:", get_text(response_tensors[-gen_len:]))

# Repeat the process for the second sample. This section generates a response for a given query, decodes the relevant
# part, and appends it to the `response_tensors` list.
query=query_tensors[1]
gen_len = output_length_sampler()
generation_kwargs["max_new_tokens"] = gen_len
response = ppo_trainer.generate(query, **generation_kwargs)
tokenizer.decode(response.squeeze()[-gen_len:], skip_special_tokens=True)
print("query:",get_text(query))
print("response ouput :", get_text(response_tensors))
response_tensors.append(response.squeeze()[-gen_len:])
print("newly generated tokens form response:", get_text(response_tensors[-gen_len:]))

batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
batch["response"]

batch




### Scoring function
texts = [q + r for q, r in zip(batch["query"], batch["response"])]
texts

pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
pipe_outputs

positive_scores = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "POSITIVE"
]
rewards = [torch.tensor(score) for score in positive_scores]
rewards


### Proximal policy optimization

print("query:", get_text(query_tensors))
print("\n")
print("response:", get_text(response_tensors))

batch_size=128
pad_token_id = tokenizer.pad_token_id

query_tensors = pad_list_to_batch_size(query_tensors, batch_size, pad_token_id)

response_tensors = pad_list_to_batch_size(response_tensors, batch_size, pad_token_id)
rewards=rewards+[torch.tensor(0) for _ in range(batch_size-len(rewards))]

#*Note: The following code is commented out to prevent the kernel from crashing due to the absence of a GPU in the
# current environment. To execute this code, please download the notebook and run it in an environment equipped with
# a GPU. Simply uncomment the code before running it.*
# stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

#The `stats` variable is a dictionary containing various statistics from the PPO training step. You can print out its
# keys using the function `print_ppo_stats`. These keys can be organized into two main categories:
# stats.keys()
# print_ppo_stats(stats, related_to_objective = True)
# print_ppo_stats(stats)
all_stats = []

sentiment = "POSITIVE"
#**Note:** Training the model on a CPU will be very time-consuming.
# You have pretrained the model using a GPU and saved it for your convenience. You can skip the training part and
# proceed to the next block of code and load the saved model. You can uncomment the below block of code to train
# the model yourself.

# for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     query_tensors = batch["input_ids"]
#     print(f"epoch {epoch}")

#     #### Get response from gpt2
#     response_tensors = []
#     for query in query_tensors:
#         gen_len = output_length_sampler()
#         generation_kwargs["max_new_tokens"] = gen_len
#         response = ppo_trainer.generate(query, **generation_kwargs)
#         response_tensors.append(response.squeeze()[-gen_len:])
#     batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

#     #### Compute sentiment score
#     texts = [q + r for q, r in zip(batch["query"], batch["response"])]
#     pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
#     positive_scores = [
#            item["score"]
#            for output in pipe_outputs
#            for item in output
#            if item["label"] == sentiment
#        ]
#    rewards = [torch.tensor(score) for score in positive_scores]

#     #### Run PPO step
#     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
#     ppo_trainer.log_stats(stats, batch, rewards)

#     all_stats.append(stats)

# # Save the model

# model_dir = "ppo-good"
# os.makedirs(model_dir, exist_ok=True)

# # Save model configuration and weights
# model_1.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)

!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/gSWo8GeztngSmzHpqX_RaQ/ppo-good.pkl
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/we8t5N-45dVq3VhxGwYRAg/ppo-good-tar.gz

# File name
file_name = "ppo-good-tar.gz"

# Open the tar.gz file
with tarfile.open(file_name, "r:gz") as tar:
    # Extract all the contents into the current directory
    tar.extractall()

print("Extraction completed.")
#Extraction completed.

model_dir = "ppov3new1"
model_1 = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load training stats
file_name = "ppo-good.pkl"
with open(file_name, 'rb') as f:
    all_stats = pickle.load(f)

model_1.to(device)

## Plotting PPO training loss and mean
#
# 1. **Extracting values**:
#     - `loss_values`: Total loss values from `all_stats`.
#     - `reward_values`: Mean reward values from `all_stats`.
#
# 2. **Plotting the loss**:
#     - Line plot of total loss over epochs.
#
# 3. **Plotting the rewards**:
#     - Line plot of mean reward over epochs.
#
# 4. **Displaying the plots**:
#     - Arrange and show the plots using `plt.tight_layout()` and `plt.show()`.

loss_values = [stat['ppo/loss/total'] for stat in all_stats]
reward_values = [stat['ppo/mean_scores'] for stat in all_stats]

# Plotting the loss
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(loss_values, label='Total Loss', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PPO Training Loss over Time')
plt.legend()
plt.grid(True)

# Plotting the rewards
plt.subplot(2, 1, 2)
plt.plot(reward_values, label='Mean Reward', color='g')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('PPO Mean Reward over Time')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

## Generating and analyzing text with PPO and reference models
# **Device Setup**:
#     - Determine if CUDA is available and set the device accordingly.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the pipeline device
pipeline_device = 0 if device.type == "cuda" else -1

# **Text generation function**:
#     - `generate_some_text(input_text, my_model)`: Tokenizes input text, generates a response, and decodes it.

gen_kwargs = {"min_length": -1, "max_new_tokens":20, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}
def generate_some_text(input_text,my_model):
# Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    generated_ids = my_model.generate(input_ids,**gen_kwargs )

    # Decode the generated text
    generated_text_ = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text_

#Generate text with PPO model: - Generate text using the PPO-trained model.
input_text = "Once upon a time in a land far"

generated_text=generate_some_text(input_text,model_1)
generated_text

# **Sentiment Analysis**:
#     - Analyze the sentiment of the generated text using `sentiment_pipe`.
pipe_outputs = sentiment_pipe(generated_text, **sent_kwargs)
pipe_outputs


# **Generate text with reference model**:
#     - Generate text using the reference model.
#
generated_text = generate_some_text(input_text,ref_model)
generated_text


## Comparing PPO and reference models on
# 1. **Generation Parameters**:
#     - Define `gen_kwargs` for text generation.
#
# 2. **Prepare Batch**:
#     - Sample a batch of size `bs` from the dataset and extract query tensors.
#
# 3. **Generate Responses**:
#     - For each query tensor, generate responses using both the reference model and the PPO model.
#
# 4. **Decode Responses**:
#     - Decode the generated response tensors into human-readable text.
#
# 5. **Compute Sentiment Scores**:
#     - Prepare texts by concatenating queries and responses.
#     - Compute sentiment scores for the responses before and after training using `sentiment_pipe`.
#
# 6. **Store Results**:
#     - Store queries, responses, and sentiment scores in `game_data`.
#     - Convert `game_data` into a DataFrame and return it.

def compare_models_on_dataset(model, ref_model, dataset, tokenizer, sentiment_pipe, sent_kwargs, device, output_length_sampler):
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    bs = 16
    game_data = dict()
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(bs)
    game_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    response_tensors_ref, response_tensors = [], []

    # Get maximum position embeddings for both models
    max_position_embeddings_ref = ref_model.config.max_position_embeddings
    max_position_embeddings_model = model.config.max_position_embeddings

    for i in range(bs):
        gen_len = output_length_sampler()

        # Convert query tensors to input IDs
        input_ids = torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device)

        # ********** Process for ref_model **********
        total_length_ref = input_ids.shape[-1] + gen_len
        if total_length_ref > max_position_embeddings_ref:
            # Truncate input_ids to fit within the max length
            max_input_length_ref = max_position_embeddings_ref - gen_len
            input_ids_ref = input_ids[:, -max_input_length_ref:]
            total_length_ref = input_ids_ref.shape[-1] + gen_len
        else:
            input_ids_ref = input_ids

        output = ref_model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
            max_new_tokens=gen_len,
            **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors_ref.append(output)

        # ********** Process for model **********
        total_length_model = input_ids.shape[-1] + gen_len
        if total_length_model > max_position_embeddings_model:
            max_input_length_model = max_position_embeddings_model - gen_len
            input_ids_model = input_ids[:, -max_input_length_model:]
            total_length_model = input_ids_model.shape[-1] + gen_len
        else:
            input_ids_model = input_ids

        output = model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
            max_new_tokens=gen_len,
            **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors.append(output)

    game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
    game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

    texts_before = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
    game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts_before, **sent_kwargs)]

    texts_after = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
    game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts_after, **sent_kwargs)]

    df_results = pd.DataFrame(game_data)
    return df_results

df_results = compare_models_on_dataset(model_1, ref_model, dataset, tokenizer, sentiment_pipe, sent_kwargs, device, output_length_sampler)
df_results

### Running the PPO model with negative sentiment
sentiment = "NEGATIVE"

# for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     query_tensors = batch["input_ids"]
#     print(f"epoch {epoch}")

#     #### Get response from gpt2
#     response_tensors = []
#     for query in query_tensors:
#         gen_len = output_length_sampler()
#         generation_kwargs["max_new_tokens"] = gen_len
#         response = ppo_trainer.generate(query, **generation_kwargs)
#         response_tensors.append(response.squeeze()[-gen_len:])
#     batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

#     #### Compute sentiment score
#     texts = [q + r for q, r in zip(batch["query"], batch["response"])]
#     pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
#     negative_scores = [
#            item["score"]
#            for output in pipe_outputs
#            for item in output
#            if item["label"] == sentiment
#        ]
#    rewards = [torch.tensor(score) for score in negative_scores]

#     #### Run PPO step
#     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
#     ppo_trainer.log_stats(stats, batch, rewards)

#     all_stats.append(stats)

# # Save the model

# model_dir = "ppo-bad"
# os.makedirs(model_dir, exist_ok=True)

# # Save model configuration and weights
# model_0.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)

#**Note:** Training the model on a CPU will be very time-consuming. The model has been pretrained using a GPU and saved
# for your convenience. You can skip the training part, proceed to the next block of code, and load the saved model.
# You can also uncomment the above training block of code to train the model yourself.
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/8zCp__SHRSgGVlf5yP50Ag/ppo-bad-tar.gz
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/jMW99Z9mvxesgYR-H6y6Yw/ppo-bad.pkl

import tarfile
# File name
file_name = "ppo-bad-tar.gz"

# Open the tar.gz file
with tarfile.open(file_name, "r:gz") as tar:
    # Extract all the contents into the current directory
    tar.extractall()

print("Extraction completed.")
#Extraction completed.

import tarfile
model_dir = "ppov3new_bad1"
model_0 = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load training stats
file_name = "ppo-bad.pkl"
with open(file_name, 'rb') as f:
    all_stats = pickle.load(f)

model_0.to(device)

### Comparing models with negative sentiment
df_results = compare_models_on_dataset(model_0, ref_model, dataset, tokenizer, sentiment_pipe, sent_kwargs, device, output_length_sampler)
df_results



### Comparing PPO models
df_results = compare_models_on_dataset(model_0, model_1, dataset, tokenizer, sentiment_pipe, sent_kwargs, device, output_length_sampler)
df_results
