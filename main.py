from eval_datasets.mmlu_pro.load_mmlu_pro import load_mmlu_pro_hy
from eval_datasets.armenian_history.load_armenian_history_test import load_armenian_history_test
from eval_datasets.armenian_language_and_literature.load_armenian_language_test import load_armenian_language_test
from eval_datasets.mathematics.load_matemathics_test import load_mathematics_test
from generate_responses import get_completions
from load_configuration import load_config, validate_model_config, validate_dataset_config
from evaluate_model import evaluation
import argparse
import yaml

def main(model_config_path, generation_config_path, dataset_config_path, get_results = True, batch_size = 8):
    if not model_config_path:
        raise ValueError("Error: 'model_config_path' is required.")

    model_config = load_config(model_config_path)
    model_config = validate_model_config(model_config)  

    if generation_config_path:
        generation_config = load_config(generation_config_path) or {}
    else:
        generation_config = {}

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

    print(f'Model Config: {model_config}')
    print(f'Generation Config: {generation_config}')
    print(f'Datasets Loaded: {list(datasets.keys())}')
    print(f'Batch Size: {batch_size}')
    print(f'Calculate results: {get_results}')

    get_completions(model_config, generation_config, batch_size, **datasets)
    
    if get_results:
        evaluation(model_config_path = model_config_path, dataset_config_path = dataset_config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on multiple datasets.")
    parser.add_argument("--config", help="Path to the YAML config file.", default='configs/config.yaml')
    parser.add_argument("--model_config", help="Path to the model configuration file.", default="configs/model_config.yaml")
    parser.add_argument("--generation_config", help="Path to the generation configuration file.", default="configs/generation_config.yaml")
    parser.add_argument("--dataset_config", help="Path to the dataset configuration file.", default=None)
    parser.add_argument("--get_results", type=bool, help="Whether to run evaluation after generating completions.", default=True)
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.", default=8)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as file:
            yaml_config = yaml.safe_load(file)

        args.model_config = yaml_config.get("model_config", args.model_config)
        args.generation_config = yaml_config.get("generation_config", args.generation_config)
        args.dataset_config = yaml_config.get("dataset_config", args.dataset_config)
        args.get_results = yaml_config.get("get_results", args.get_results)
        args.batch_size = yaml_config.get("batch_size", args.batch_size)

    main(
        model_config_path=args.model_config,
        generation_config_path=args.generation_config,
        dataset_config_path=args.dataset_config,
        get_results=args.get_results,
        batch_size=args.batch_size
    )