import torch
import vllm
from utils import load_with_automodel, get_automodel_completion, load_with_vllm, get_vllm_completion
from tqdm import tqdm
import os
import pandas as pd
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_completions(model_config, generation_config, batch_size, **kwargs):
    model_name = model_config['model_name']

    print(f"Starting evaluation for model: {model_name}")
    model_name = model_name.replace('/', '_')

    if model_config['vllm_optimisation'] == True:
        llm = {'local': load_with_vllm(model_config)}
    elif model_config['vllm_optimisation'] == False:
        llm = {'local': load_with_automodel(model_config, device)}

    for idx, (df_name, df) in enumerate(kwargs.items()):
        outputs_file = f"model_outputs/{model_name}/{df_name}_output.json"
        generate_and_process_dataset_completions(df, llm, model_config, generation_config, outputs_file, batch_size) 

def generate_and_process_dataset_completions(df, llm, model_config, generation_config, outputs_file, batch_size):
    model_name = model_config['model_name']
    model_name = model_name.replace('/', '_')
    outputs_dir = f"model_outputs/{model_name}"
    batch_results = []

    os.makedirs(outputs_dir, exist_ok=True)

    if os.path.exists(outputs_file):
        with open(outputs_file, 'r') as f:
            batch_results = json.load(f)

    start_index = len(batch_results)

    for i in tqdm(range(start_index, len(df), batch_size), desc="Evaluating batches"):
        batch = df.iloc[i:i + batch_size]
        true_answers = batch['label'].tolist()
        system_prompts = batch['system_prompt'].tolist()
        inputs = batch['input_prompt'].tolist()

        if isinstance(list(llm.values())[0], vllm.entrypoints.llm.LLM):
            model = list(llm.values())[0]
            outputs = get_vllm_completion(model, generation_config, system_prompts, inputs)
        else: 
            model = list(llm.values())[0][0]
            tokenizer = list(llm.values())[0][1]
            outputs = get_automodel_completion(model, tokenizer, generation_config, system_prompts, inputs, device)

        batch_info = []
        for idx, (text, true_answer) in enumerate(zip(outputs, true_answers)):
            try:
                batch_info.append({
                    'index': i + idx,
                    'system_prompt': system_prompts[idx],
                    'input': inputs[idx],
                    'output': text,
                    'label': true_answers[idx].tolist()
                })
            except:
                batch_info.append({
                    'index': i + idx,
                    'system_prompt': system_prompts[idx],
                    'input': inputs[idx],
                    'output': text,
                    'label': true_answers[idx]
                })

        batch_results.extend(batch_info)
        with open(outputs_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=4, ensure_ascii=False)
