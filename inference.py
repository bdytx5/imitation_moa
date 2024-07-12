from typing import List, Tuple
from utils import set_seed
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import wandb

wandb.login(key="82cbd27eead1f27bb5cc79b0a83a3a70fd4595f0")

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

def inference(model: str,  prompt: str) -> str:
    """Runs inference on a model given a prompt. Returns the string output."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
    )
    model_input = tokenizer(prompt, return_tensors="pt").to(device)

    _ = model.eval()
    with torch.no_grad():
        out = model.generate(**model_input, max_new_tokens=100)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def compare_inference(base_model: str, trained_model: str, inputs: List[str]) -> Tuple[List[str], List[str]]:
    """Generates inference across the list of inputs for a base and a trained model."""
    base_model_outputs = []
    for x in inputs:
        output = inference(base_model, x)
        base_model_outputs.append(output)

    trained_model_outputs = []
    for x in inputs:
        output = inference(trained_model, x)
        trained_model_outputs.append(output)

    return base_model_outputs, trained_model_outputs

def create_wandb_table(wandb_project_name: str, inputs: str, base_model_outputs: List[str], trained_model_outputs: List[str]):
    run = wandb.init(
        project=wandb_project_name,  # Project name.
        name="log_dataset",          # name of the run within this project.
        config={                     # Configuration dictionary.
            "split": "test"
        },
    ) 

    data = []
    for x, y, z in zip(inputs, base_model_outputs, trained_model_outputs):
        data.append([x, y, z])

    table = wandb.Table(data=data, columns=["input", "base_model_output", "trained_model_output"])
    run.log({"compare_table": table})
    run.finish()