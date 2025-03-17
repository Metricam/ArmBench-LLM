from datasets import load_dataset, Dataset
import pandas as pd
import os
import re

categories_dict = {
    'biology': 'կենսաբանություն',
    'engineering': 'ինժեներիա',
    'math': 'մաթեմատիկա',
    'history': 'պատմություն',
    'philosophy': 'փիլիսոփայություն',
    'health': 'առողջապահություն',
    'computer science': 'համակարգչային գիտություն',
    'economics': 'տնտեսագիտություն',
    'other': 'այլ',
    'physics': 'ֆիզիկա',
    'chemistry': 'քիմիա',
    'psychology': 'հոգեբանություն',
    'law': 'իրավաբանություն',
    'business': 'բիզնես'
}

def mmlu_pro_hy_processing(row):
    system_prompt_path = 'eval_datasets/mmlu_pro/system_prompt.md'
    with open(system_prompt_path, 'r') as file:
        system_prompt = file.read()

    input_prompt_path = 'eval_datasets/mmlu_pro/input_prompt.md'
    with open(input_prompt_path, 'r') as file:
        input_prompt = file.read()

    system_prompt = system_prompt.format(category = row['category'])
    formatted_input_prompt = input_prompt.format(question = row['question_arm'])+'\n'

    choice_map = "ABCDEFGHIJ"
    options = row['options_arm']

    for i, opt in enumerate(options):
        formatted_input_prompt += "({}) {}\n".format(choice_map[i], opt)
    formatted_input_prompt += "Output: "
    return pd.Series({'system_prompt': system_prompt, 'input_prompt': formatted_input_prompt})

def load_mmlu_pro_hy():
    df = load_dataset('Metric-AI/mmlu-pro-hy')['validation'].to_pandas()
    df = df.rename(columns={'answer': 'label'})
    inputs = df.apply(mmlu_pro_hy_processing, axis=1)
    df = pd.concat([df, inputs], axis=1)
    return df
