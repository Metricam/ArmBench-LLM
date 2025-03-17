from datasets import load_dataset
import pandas as pd
import os
from datasets import Dataset
import re

cpu_count = os.cpu_count()

def mathematics_test_processing(row):
    if row['task_type'] == 6:
        system_prompt_path = 'eval_datasets/mathematics/task1_system_prompt.md'
        input_prompt_path = 'eval_datasets/mathematics/task1_input_prompt.md'
    elif row['task_type'] == 7:
        system_prompt_path = 'eval_datasets/mathematics/task2_system_prompt.md'
        input_prompt_path = 'eval_datasets/mathematics/task2_input_prompt.md'
    elif row['task_type'] == 3:
        system_prompt_path = 'eval_datasets/mathematics/task3_system_prompt.md'
        input_prompt_path = 'eval_datasets/mathematics/task3_input_prompt.md'

    with open(system_prompt_path, 'r') as file:
        system_prompt = file.read()
    with open(input_prompt_path, 'r') as file:
        input_prompt = file.read()

    formatted_input_prompt = input_prompt.format(task=row['task'], question=row['question'])
    if row['task_type'] !=7:
        choice_map = "ABCDEFGHIJ"  
        for i in range(len(row['choices'])):
            formatted_input_prompt += "{}. {}\n".format(choice_map[i], row['choices'][i])
    
    formatted_input_prompt += "Output: "
    return pd.Series({'system_prompt': system_prompt, 'input_prompt': formatted_input_prompt})

def load_mathematics_test():
    df = load_dataset('Metric-AI/mathematics-test-2025-1')['train'].to_pandas()
    inputs = df.apply(mathematics_test_processing, axis=1)
    df = pd.concat([df, inputs], axis=1)
    return df
