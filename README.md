# Arm-LLM-Benchmark

A comprehensive Armenian model evaluation framework for benchmarking large language models (LLMs).

## Overview

Arm-LLM-Benchmark enables thorough evaluation of language models across multiple Armenian-specific and general datasets. The framework supports both VLLM optimization and Hugging Face's AutoModelForCausalLM for flexible evaluation options.

## Features

- Evaluate models on specialized Armenian datasets (language, literature, history)
- Support for the Armenian version of MMLU-Pro (MMLU-Pro-Hy)
- vLLM optimization for faster inference
- Ability to resume interrupted evaluations
- Comprehensive scoring system
- Submit your model results to the [Arm-LLM-Benchmark](https://huggingface.co/spaces/Metric-AI/Arm-LLM-Benchmark) leaderboard

## Getting Started

### Prerequisites
- Python 3.10.16
- CUDA-compatible GPU (preferable)
- Required dependencies (install via `pip install -r requirements.txt`)

### Hardware Requirements
- **GPU Preferable**: This benchmark uses vLLM for optimization which is optimized for CUDA-compatible GPUs.

### Configuration

The benchmark already includes configuration files in the `configs` directory:

1. `model_config.yaml` - Contains model specification and parameters
2. `generation_config.yaml` - Contains text generation parameters
3. `dataset_config.yaml` - Contains dataset selection for evaluation

You can modify these existing config files according to your needs before running the benchmark.

#### General Configuration Example

Example of `configs/config.yaml`:

```yaml
model_config: "configs/model_config.yaml"
generation_config: "configs/generation_config.yaml"
dataset_config: "configs/dataset_config.yaml"
get_results: True
batch_size: 8
```

#### Model Configuration Example

Example of `configs/model_config.yaml`:

```yaml
model_name: 'your_hf_model_name'  # Required
vllm_optimisation: True  # Optional, defaults to True
# Add other model parameters supported by vLLM or Hugging Face
# Examples:
# dtype: 'bfloat16'
# trust_remote_code: True
# revision: 'main'
```

#### Generation Configuration Example

Example of `configs/generation_config.yaml`:

```yaml
# Text generation parameters
temperature: 0.7
top_p: 0.9
top_k: 40
max_tokens: 512
# Add other generation parameters according to vLLM or Hugging Face documentation
```

#### Dataset Configuration Example

Example of `configs/dataset_config.yaml`:

```yaml
datasets:
  - Armenian language and literature
  - Armenian history
  - Mathematics
  - mmlu_pro
```

If not provided, the framework will evaluate using all available datasets.

## Running Evaluations

### Configuration Files Overview

The benchmark includes the following configuration files:

1. **General configuration file**:
   - `config.yaml` references the individual configuration files and sets other parameters

2. **Individual configuration files**:
   - `model_config.yaml` - Contains model specification and parameters
   - `generation_config.yaml` - Contains text generation parameters
   - `dataset_config.yaml` - Contains dataset selection for evaluation

### Step 1: Run the main evaluation script

The main script supports command-line arguments for flexible configuration:

```bash
python main.py [--config CONFIG_PATH] [--model_config MODEL_CONFIG_PATH] 
               [--generation_config GEN_CONFIG_PATH] [--dataset_config DATASET_CONFIG_PATH] 
               [--get_results GET_RESULTS] [--batch_size BATCH_SIZE]
```

#### Command-line Arguments:

- `--config`: Path to the general YAML config file (default: 'configs/config.yaml')
- `--model_config`: Path to the model configuration file (default: 'configs/model_config.yaml')
- `--generation_config`: Path to the generation configuration file (default: 'configs/generation_config.yaml')
- `--dataset_config`: Path to the dataset configuration file (default: None)
- `--get_results`: Whether to run evaluation after generating completions (default: True)
- `--batch_size`: Batch size for processing (default: 8)

#### Examples:

**Using the general config file:**
```bash
python main.py --config configs/config.yaml
```

**Using individual configuration files:**
```bash
python main.py --model_config configs/model_config.yaml \
               --generation_config configs/generation_config.yaml \
               --batch_size 4
```

#### `get_results` Parameter

The `get_results` parameter controls whether the evaluation results are computed and saved automatically after running the main script (`main.py`).

- **`get_results=True` (default)**:  
  If set to `True`, the script will run the evaluation and generate a `results.json` file in the model's output directory automatically.

- **`get_results=False`**:  
  If set to `False`, the main script will not calculate and save the evaluation results immediately. Instead, you can manually compute the results later by running the `evaluate_model.py` script.
  
### Step 2: Evaluate model 

This step is only required if you set `get_results=False` in the previous step. Run the evaluation script with:

```bash
python evaluate_model.py [--config CONFIG_PATH] \
                         [--model_config MODEL_CONFIG_PATH] \
                         [--dataset_config DATASET_CONFIG_PATH]
```

#### Command-line Arguments:

- `--config`: Path to the general YAML config file (default: 'configs/config.yaml')
- `--model_config`: Path to the model configuration file (default: 'configs/model_config.yaml')
- `--dataset_config`: Path to the dataset configuration file (default: None)

**Important:** You must use the same configuration values in `evaluate_model.py` as you used in `main.py` to ensure consistent evaluation results.

This will create a `results.json` file in the model's output directory with comprehensive scores.

### Step 3: Submit your results to the leaderboard

After evaluating your model, you can submit your results to the [Arm-LLM-Benchmark](https://huggingface.co/spaces/Metric-AI/Arm-LLM-Benchmark) leaderboard by following these steps:

1. Push your model and tokenizer to the Hugging Face Hub
2. Add the `Arm-LLM-Benchmark` tag to your model repository
3. Include the `results.json` file in your repository

**Important Submission Requirements:**
To be included in the official ArmBench leaderboard, your evaluation **must** include at least one of the following:
- All unified exam datasets (Armenian language and literature, Armenian history, Mathematics)
- The MMLU-Pro dataset (mmlu_pro)

Results that include only partial evaluations (e.g., only Armenian language and history tests) will be discarded from the Hugging Face ArmBench space.

Here's an example of how to push your model with results to the Hugging Face Hub:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from huggingface_hub import HfApi

hub_repo_name = "company_name/model_name"
results_path = "results.json"

model = AutoModelForCausalLM.from_pretrained(hub_repo_name)
tokenizer = AutoTokenizer.from_pretrained(hub_repo_name)

api = HfApi()
api.upload_file(
    path_or_fileobj=results_path,
    path_in_repo="results.json",
    repo_id=hub_repo_name,
    repo_type="model",
    commit_message="Add ARM benchmark results"
)
model.push_to_hub(hub_repo_name, tags=['Arm-LLM-Benchmark'])
tokenizer.push_to_hub(hub_repo_name, tags=['Arm-LLM-Benchmark'])
```

After pushing your model with the `Arm-LLM-Benchmark` tag and `results.json` file that meets the submission requirements, your model will appear on the ArmBench leaderboard.

## Output Structure

During evaluation, the framework creates output directories in the following structure:
```
model_outputs/
└── {model_name}/
    ├── dataset_x_outputs.json
    └── results.json
```

Where `{model_name}` is the model name specified in your `model_config.yaml` file. Each model gets its own directory to store:
- Raw model responses for each dataset
- Final `results.json` with detailed performance metrics

The `model_outputs` directory also serves as a valuable resource for error analysis, allowing you to inspect the model's raw responses and understand its performance characteristics in detail.

### Handling Interrupted Evaluations

If an evaluation is interrupted without changing parameters, you can resume it by running the main script again with the same configuration parameters.

**Important:** If you need to change any parameters during evaluation (model configuration, generation settings, etc.), you MUST delete the corresponding model directory from the `model_outputs` folder before continuing. For example, if you want to change parameters for a model named "google_gemma-2-2b-it", you should delete the entire `model_outputs/google_gemma-2-2b-it` directory.

It is strongly recommended to complete the entire evaluation with the same set of parameters. This ensures that all evaluation data for a model is generated with consistent parameters. Failing to delete previous results when changing parameters may result in inconsistent evaluation results where some parts of the dataset were evaluated with different parameters than others.

## Available Datasets

This benchmark evaluates Language Models on Armenian-specific tasks, including Armenian Unified Test Exams and a subsample of the MMLU-Pro benchmark, translated into Armenian (MMLU-Pro-Hy). It is designed to measure the models' understanding and generation capabilities in the Armenian language.

The framework includes the following datasets:
- Armenian language and literature
- Armenian history
- Mathematics
- mmlu_pro

You can evaluate on specific datasets by specifying them in the `dataset_config.yaml` file.

## Advanced Configuration

### vLLM Optimization

By default, the framework uses vLLM for optimized inference. This requires a CUDA-compatible GPU.

### Model Parameters

You can add any parameter supported by vLLM or Hugging Face's AutoModelForCausalLM to the model configuration file. See their respective documentation for details:
- [vLLM Documentation](https://docs.vllm.ai/en/stable/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/main_classes/model)

### Generation Parameters

Similarly, you can customize text generation by adding parameters to the generation configuration file:
- For vLLM: See [vLLM Generation Parameters](https://vllm.readthedocs.io/en/latest/serving/engine_args.html)
- For Hugging Face: See [Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)

## Leaderboard

Check out the current leaderboard at [Metric-AI/Arm-LLM-Benchmark](https://huggingface.co/spaces/Metric-AI/Arm-LLM-Benchmark) to see how your model compares to others on Armenian language tasks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
