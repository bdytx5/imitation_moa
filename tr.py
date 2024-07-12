

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    TrainerCallback,
)
from trl import SFTTrainer
import wandb
import json
import random 
# import os
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

# Login to Weights and Biases

# wandb.init(project="moa", entity='byyoung3')
# Seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Configuration
# model_name = "sshleifer/tiny-gpt2"
model_name = "microsoft/Phi-3-mini-128k-instruct"
# model_name = "microsoft/Phi-3-mini-4k-instruct"
max_seq_length = 1024
output_dir = "./results"
num_train_epochs = 1
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 32
learning_rate = 5e-6
logging_steps = 10
save_steps = 10
eval_steps = 10
warmup_steps = 0
save_total_limit = 1  # will save best and latest 
train_file_path = './final_ds/train_completions.jsonl'
val_file_path = './final_ds/test_completions.jsonl'
loss_exploded = False  
gradient_checkpointing = True 
precision = "fp16"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def load_jsonl_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

train_data = load_jsonl_data(train_file_path)
val_data = load_jsonl_data(val_file_path)

# Convert data to Hugging Face Dataset format
data_list_train = [dict(d) for d in train_data]
data_list_val = [dict(d) for d in val_data]

train_dataset = Dataset.from_list(data_list_train)
val_dataset = Dataset.from_list(data_list_val)

# Filter examples based on max_seq_length
def filter_examples(example):
    combined_text = example['input']
    tokens = tokenizer.encode(combined_text)
    return len(tokens) < max_seq_length

train_dataset = train_dataset.filter(filter_examples)
val_dataset = val_dataset.filter(filter_examples)



# Format chat template

# <|system|>
# You are a helpful assistant.<|end|>
# <|user|>
# Question?<|end|>
# <|assistant|>
def format_chat_template(example):
    return {'text': f"<|system|>You are a helpful assistant.<|end|>\n<|user|>{example['input']}<|end|>\n<|assistant|><|{example['model_name']}|>\n{example['output']}\n<|end|>"}

# Format and prepare datasets
train_dataset = train_dataset.map(format_chat_template)
val_dataset = val_dataset.map(format_chat_template)

print(f"Number of examples in the train set: {len(train_dataset)}")
print(f"Number of examples in the validation set: {len(val_dataset)}")

def create_and_prepare_model():
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = create_and_prepare_model()

training_arguments = TrainingArguments(
    num_train_epochs=num_train_epochs,
    gradient_checkpointing=gradient_checkpointing,
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_total_limit=save_total_limit,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=precision == "fp16",
    bf16=precision == "bf16",
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    warmup_steps=warmup_steps,
    lr_scheduler_type="linear",
    report_to='wandb',
    save_steps=save_steps,
    save_strategy="steps",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end = True,
    deepspeed="./z3.json",
)



print("training")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,

)
print("training")

trainer.train()



model.save_pretrained("results/BEST_MODEL")
