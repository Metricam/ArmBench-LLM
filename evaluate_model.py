from score_calculation import calculate_scores_for_mmlu, calculate_scores_for_exam_tests
from load_configuration import load_config, validate_model_config, validate_dataset_config
from eval_datasets.mmlu_pro.load_mmlu_pro import load_mmlu_pro_hy
from eval_datasets.armenian_history.load_armenian_history_test import load_armenian_history_test
from eval_datasets.armenian_language_and_literature.load_armenian_language_test import load_armenian_language_test
from eval_datasets.mathematics.load_matemathics_test import load_mathematics_test
from tqdm import tqdm
import os
import pandas as pd
import json
import argparse
import yaml

def evaluation(model_config_path = None, dataset_config_path = None):
    results_json = {"mmlu_results": [], "unified_exam_results": []}
    
    if not model_config_path:
        raise ValueError("Error: 'model_config_path' is required.")

    model_config = load_config(model_config_path)
    model_config = validate_model_config(model_config)
    model_name = model_config['model_name'] 
    model_name = model_name.replace('/', '_')

    dataset_loaders = {
        'Armenian language and literature': load_armenian_language_test,
        'Armenian history': load_armenian_history_test,
        'Mathematics': load_mathematics_test,
        'mmlu_pro': load_mmlu_pro_hy
    }

    if dataset_config_path:
        dataset_config = load_config(dataset_config_path)
        dataset_list = validate_dataset_config(dataset_config)

        if not dataset_list:
            dataset_list = list(dataset_loaders.keys())
    else:
        dataset_list = list(dataset_loaders.keys())

    datasets = {name: dataset_loaders[name]() for name in dataset_list}

    unified_exam_scores = []

    for idx, (df_name, df) in enumerate(datasets.items()):
        outputs_file = f"model_outputs/{model_name}/{df_name}_output.json"

        if df_name == 'mmlu_pro':
            results = calculate_scores_for_mmlu(outputs_file, df, model_name)
            for category, score in results.items():
                if category != "Model":  
                    results_json["mmlu_results"].append({"category": category, "score": round(score, 4)})

        else:
            results = calculate_scores_for_exam_tests(outputs_file, df, model_name) 
            unified_exam_scores.append(results["score"])
            results_json["unified_exam_results"].append({
                "category": df_name,
                "score": round(results["score"], 4)
            })  

    if len(unified_exam_scores) == 3:
        average_score = sum(unified_exam_scores)/len(unified_exam_scores)
        results_json["unified_exam_results"].append({
                "category": 'Average',
                "score": round(average_score, 4)
            })

    json_output_path = f"model_outputs/{model_name}/results.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on multiple datasets.")
    parser.add_argument("--config", help="Path to the YAML config file.", default='configs/config.yaml')
    parser.add_argument("--model_config", help="Path to the model configuration file.", default="configs/model_config.yaml")
    parser.add_argument("--dataset_config", help="Path to the dataset configuration file.", default=None)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as file:
            yaml_config = yaml.safe_load(file)

        args.model_config = yaml_config.get("model_config", args.model_config)
        args.dataset_config = yaml_config.get("dataset_config", args.dataset_config)

    evaluation(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
    )
