import subprocess
import torch
from datasets import load_dataset
import json
import ollama

class ModelInference:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate_one_completion(self, prompt):
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            completion = response['message']['content']
            return {"completion": completion}
        except Exception as e:
            print(f"Error with model {self.model_name}: {str(e)}")
            return {"completion": f"Failed to generate response with {self.model_name}"}

# List of models to iterate over
models = [
    'phi3', 'llama2', 'llama3', 'mistral', 'mistral:instruct',
    'llama3:instruct', 'gemma', 'gemma:instruct'
]

# Load the dataset from Hugging Face
dataset = load_dataset('flytech/python-codes-25k')

# Process each model
for model_name in models:
    try:
        # Pull model using subprocess
        subprocess.run(['ollama', 'pull', model_name], check=True)
        print(f"Model {model_name} pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull model {model_name}: {e}")
        continue  # Skip to the next model if pulling fails

    model_inf = ModelInference(model_name)
    output_file = f'generated_completions_{model_name.replace(":", "_")}.jsonl'

    with open(output_file, 'w') as outfile:
        # for entry in dataset['train'].select(range(5)):  # Adjust dataset split as needed\
        for entry in dataset['train']:  # Adjust dataset split as needed\
            print(type(entry))
            instruction = "write a python script to accomplish the following task:" + str(entry['instruction']) + " just write the code formatted as following!!!!!!!!!->```python THE_CODE ``` OK NOW WRITE THE CODE! Just CODE NOTHING ELSE!!!!!!"
            output = model_inf.generate_one_completion(instruction)
            id_ = ''.join(e for e in instruction if e.isalnum())  # Create a simple alphanumeric ID
            print(output)
            # Create and write the JSON entry
            json_entry = {
                'id': id_,
                'input': instruction,
                'output': output['completion'],
                'gt': entry['output']
            }
            json.dump(json_entry, outfile)

            outfile.write('\n')


    try:
        # Pull model using subprocess
        subprocess.run(['ollama', 'rm', model_name], check=True)
        subprocess.run(['rm', '-rf', "~/.ollama/models/blobs/"], check=True)  
        print(f"Model {model_name} rm successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to rm model {model_name}: {e}")
        continue  # Skip to the next model if pulling fails

    print(f"Completions generated and saved for model {model_name}.")

print("All model processing completed.")

