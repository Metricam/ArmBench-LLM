import yaml
import json
import warnings

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_model_config(config):
    if config is None:
        raise ValueError("Model config cannot be None or an empty file")
    if "model_name" not in config:
        raise ValueError("Model config must contain 'model_name'")
    if "vllm_optimisation" not in config:
        config["vllm_optimisation"] = True
    return config

def validate_dataset_config(config):
    if config is None:
        raise ValueError("Dataset config cannot be None or an empty file")
    valid_datasets = {
        "Armenian language and literature",
        "Armenian history",
        "Mathematics",
        "mmlu_pro"
    }

    if "datasets" not in config or not isinstance(config["datasets"], list):
        raise ValueError("Dataset config must contain a non-empty list under the 'datasets' key.")

    if not config["datasets"]:
        raise ValueError("Dataset list cannot be empty.")

    invalid_datasets = [ds for ds in config["datasets"] if ds not in valid_datasets]
    if invalid_datasets:
        raise ValueError(f"Invalid datasets found: {invalid_datasets}. Valid options are: {valid_datasets}")

    return config["datasets"]

    
