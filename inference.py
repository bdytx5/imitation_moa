##### t0d0 
#### -> add inference for stock phi3 on val set 
########## -> log all results to wandb tables in a way thats easily comparable (eg same q's grouped together) (including original results from val dataset (group by id key in dataset) ) 
### right now we arent using any metrics, mainly just logging results. We can do this later, so possibly log each models results to a jsonl file as well for easy comparision later 
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    TrainerCallback,
    pipeline
)
from trl import SFTTrainer
import wandb
import json
import random


# Seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Configuration
model_name = "bdytx5/imm_moa_phi3_128k_instruct"
max_seq_length = 1024
train_file_path = './final_ds/train_completions.jsonl'
val_file_path = './final_ds/test_completions.jsonl'

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

def load_jsonl_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

val_data = load_jsonl_data(val_file_path)

# Convert data to Hugging Face Dataset format
data_list_val = [dict(d) for d in val_data]
val_dataset = Dataset.from_list(data_list_val)

# Filter examples based on max_seq_length
def filter_examples(example):
    combined_text = example['input']
    tokens = tokenizer.encode(combined_text)
    return len(tokens) < max_seq_length

val_dataset = val_dataset.filter(filter_examples)

# Format chat template
def format_chat_template(example):
    return {'text': f"You are a helpful assistant.\n{example['input']}\n<|{example['model_name']}|>\n{example['output']}\n"}

# Format and prepare datasets
train_dataset = train_dataset.map(format_chat_template)
val_dataset = val_dataset.map(format_chat_template)

print(f"Number of examples in the train set: {len(train_dataset)}")
print(f"Number of examples in the validation set: {len(val_dataset)}")

# Simple inference script
# Load the model and tokenizer for inference
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",  # Use CPU for inference
    torch_dtype="auto",
    trust_remote_code=True
)

# Initialize the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define generation arguments
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Run inference on the validation set
def generate_output(example):
    messages = example['text']
    output = pipe(messages, **generation_args)
    example['generated_text'] = output[0]['generated_text']
    print(example)
    return example

# Apply the inference to the validation set
val_dataset = val_dataset.map(generate_output)
