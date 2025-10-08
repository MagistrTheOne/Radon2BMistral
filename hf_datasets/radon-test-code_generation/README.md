# radon-test-code_generation

## Description
RADON test dataset for code_generation evaluation

## Tags
test, evaluation, code_generation

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("MagistrTheOne/radon-test-code_generation")
```

## Examples
```python
# Load and use the dataset
data = dataset['train']
for example in data:
    print(example)
```
