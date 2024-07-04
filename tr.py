from hf_olmo import *  # registers the Auto* classes
import os
import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import wandb
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

###### FOR MultiGPu training https://stackoverflow.com/questions/76675018/how-does-one-use-accelerate-with-the-hugging-face-hf-trainer
# Login to Weights and Biases
wandb.login(key="82cbd27eead1f27bb5cc79b0a83a3a70fd4595f0")

# Seed for reproducibility
torch.manual_seed(42)

# Configuration
model_name = "allenai/OLMo-7B-SFT"
max_seq_length = 2048
output_dir = "./easy_align_results"
num_train_epochs = 1
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 32
learning_rate = 5e-6
logging_steps = 5
save_steps = 1000
eval_steps = 1000
warmup_steps = 10
save_total_limit = 5  # Number of checkpoints to keep
train_file_path = 'combined_completions_train.jsonl'
val_file_path = 'combined_completions_val.jsonl'


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load and filter dataset
def load_and_filter_dataset(train_file_path, val_file_path, test_ratio=0.1):
    # Load dataset from JSONL file
    train_dataset = load_dataset('json', data_files=train_file_path, split='train')
    val_dataset = load_dataset('json', data_files=val_file_path, split='train')

    # Filter examples based on max_seq_length
    def filter_examples(example):
        combined_text = example['input']
        tokens = tokenizer.encode(combined_text)
        return len(tokens) < max_seq_length

    filtered_train_dataset = train_dataset.filter(filter_examples)
    filtered_val_dataset = val_dataset.filter(filter_examples)

    # Shuffle and split dataset
    shuffled_train_dataset = filtered_train_dataset.shuffle(seed=42)
    train_test_dataset = shuffled_train_dataset.train_test_split(test_size=test_ratio)

    return train_test_dataset['train'], train_test_dataset['test'], filtered_val_dataset

# Format chat template
def format_chat_template(example):
    return {'text': f"<|user|>\n{example['input']}\n<|model_name|>\n{example['model_name']}\n<|output|>\n{example['output']}\n"}

# Load and process datasets

train_dataset, test_dataset, val_dataset = load_and_filter_dataset(train_file_path, val_file_path)

# Format and prepare datasets
train_dataset = train_dataset.map(format_chat_template)
test_dataset = test_dataset.map(format_chat_template)
val_dataset = val_dataset.map(format_chat_template)

print(f"Number of examples in the train set: {len(train_dataset)}")
print(f"Number of examples in the test set: {len(test_dataset)}")
print(f"Number of examples in the validation set: {len(val_dataset)}")

# Create and prepare model
def create_and_prepare_model():
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["att_proj", "attn_out", "ff_proj", "ff_out"]
    )
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:0')
    lora_model = get_peft_model(model, peft_config)

    tokenizer.pad_token = tokenizer.eos_token
    return lora_model, peft_config, tokenizer

model, peft_config, tokenizer = create_and_prepare_model()

training_arguments = TrainingArguments(
    num_train_epochs=num_train_epochs,
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim="paged_adamw_8bit",
    save_total_limit=save_total_limit,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=False,
    bf16=True,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    warmup_steps=warmup_steps,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to='wandb',
    save_steps=save_steps,
    save_strategy="steps",
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
)

trainer.train()

model.save_pretrained("ez/olmo-sft-ez")
