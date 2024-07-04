from datasets import load_dataset
from sympy import subsets

# full dataset (file IDs only)
ds = load_dataset("bigcode/the-stack-v2", split="train")

# specific language (e.g. Dockerfiles)
ds = load_dataset("bigcode/the-stack-v2", "Python", split="train")

# dataset streaming (will only download the data as needed)
ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
for sample in iter(ds): 
    print(sample) 