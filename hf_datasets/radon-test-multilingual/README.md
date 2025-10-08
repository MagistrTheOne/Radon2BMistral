# radon-test-multilingual

## Description
RADON test dataset for multilingual evaluation

## Tags
test, evaluation, multilingual

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("MagistrTheOne/radon-test-multilingual")
```

## Examples
```python
# Load and use the dataset
data = dataset['train']
for example in data:
    print(example)
```
