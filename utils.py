import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
import time
import os

def load_with_automodel(model_config, device):
    model_name = model_config['model_name']
    config = model_config.copy()
    del config['model_name']
    del config['vllm_optimisation']
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        **config
        )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer
    
def get_automodel_completion(model, tokenizer, generation_config, system_prompts, texts, device):
    model.eval()
    system_and_texts = [system_prompts[i]+'\n'+texts[i] for i in range(len(texts))]
    with torch.no_grad(): 
        try:
            inputs = tokenizer(system_and_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        except ValueError as e:
            if "pad token" in str(e):  
                tokenizer.pad_token = tokenizer.eos_token 
                inputs = tokenizer(system_and_texts, return_tensors='pt', padding=True, truncation=True).to(device)
  
        outputs = model.generate(**inputs, **generation_config)
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [decoded_text[len(input_text):].strip() for input_text, decoded_text in zip(system_and_texts, decoded_texts)]
    return generated_texts

def load_with_vllm(model_config):
    config = model_config.copy()
    del config['model_name']
    del config['vllm_optimisation']
    model_name = model_config['model_name']
    model = LLM(model=model_name, **config)
    return model

def get_vllm_completion(model, generation_config, system_prompts, texts):
    sampling_params = SamplingParams(
        **generation_config 
    )
    system_and_texts = [system_prompts[i]+'\n'+texts[i] for i in range(len(texts))]

    outputs = model.generate(system_and_texts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    return generated_texts