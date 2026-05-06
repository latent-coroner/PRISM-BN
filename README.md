# PRISM-BN: A Large-Scale Corpus of Parameterized Bayesian Networks Paired with Natural Language

A comprehensive dataset builder and evaluation framework for extracting and validating Bayesian Networks from text using Large Language Models. This repository contains tools for probabilistic graphical model construction and comparative evaluation across multiple LLM backends.
<img width="1693" height="929" alt="subgraph_generation-3" src="https://github.com/user-attachments/assets/796decaa-8dc9-4cd5-bb4d-465a75fa4cc7" />

## Overview

This project implements a pipeline for:
1. **Extracting Bayesian Networks** from natural language text using LLMs
2. **Evaluating extraction quality** through a 4-pass judgment process
3. **Comparing performance** across different LLM models
4. **Building probabilistic graphical models** with nodes, states, edges, and conditional probability distributions (CPDs)

### Key Components

- **Extraction Pipeline**: Multi-pass extraction of Bayesian Network components (nodes, states, edges, CPD matrices)
- **Evaluation Framework**: Automated quality assessment using a judge model
- **Model Comparison**: Standardized evaluation across multiple LLM backends


---

## Quick Start

### Prerequisites

- Python 3.8+
- Required Python packages:
  ```bash
  pip install anthropic huggingface-hub networkx wikipedia requests
  ```

### Environment Setup

Before running any scripts, configure your LLM API credentials. Create a `.env` file or set environment variables:

```bash
# Required API credentials
export HF_TOKEN="your_huggingface_token"
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Model configuration
export EXTRACTION_MODEL="your_extraction_model_name"
export JUDGE_MODEL="your_judge_model_name"
```

### Example Usage

```bash
# Extract Bayesian Network from text
python extractor.py --input data.json --output extracted_bn.jsonl

# Evaluate extraction quality with different models
python claude_gen_llama_eval.py --input data.json
python deepseek_gen_llama_eval.py --input data.json
python gpt_gen_llama_eval.py --input data.json
python llama_gen_llama_eval.py --input data.json
python qwen_gen_llama_eval.py --input data.json
python gemma_gen_llama_eval.py --input data.json

# Build final dataset
python build_final.py --input extracted_bn.jsonl --output final_dataset.json

# Generate subgraph analysis
python generate_subgraphs.py --input final_dataset.json
python generate_subgraph_text.py --input subgraphs.json --output subgraph_text.json

# Clean and compute CPDs
python clean_bn.py --input final_dataset.json --output cleaned_dataset.json
python cpd_calculator.py --input cleaned_dataset.json --output cpd_results.json
```

---

## Project Structure

### Core Scripts

#### **Data Extraction**
- **`extractor.py`**: Main Bayesian Network extractor
  - Implements 4-pass extraction pipeline
  - Pass 1: Node identification from text
  - Pass 2: State enumeration for each node
  - Pass 3: Edge (causal relationship) extraction
  - Pass 4: CPD (Conditional Probability Distribution) computation
  - Outputs: `bn_marginal_raw.jsonl`

#### **Evaluation & Comparison**
- **`claude_gen_llama_eval.py`**: Evaluation with Claude as extraction model
- **`deepseek_gen_llama_eval.py`**: Evaluation with DeepSeek as extraction model
- **`gpt_gen_llama_eval.py`**: Evaluation with GPT-4o-mini as extraction model
- **`llama_gen_llama_eval.py`**: Evaluation with Llama 3.3 70B as extraction model
- **`qwen_gen_llama_eval.py`**: Evaluation with Qwen 3-30B as extraction model
- **`gemma_gen_llama_eval.py`**: Evaluation with Gemma 2 9B as extraction model

All evaluation scripts implement the same 4-pass judgment process:
- **Pass 1**: Node extraction evaluation (F1 score, % matched, extra nodes)
- **Pass 2**: State extraction evaluation (F1 score, % matched)
- **Pass 3**: Edge extraction evaluation (% correct, % spurious connections)
- **Pass 4**: CPD matrix evaluation (KL divergence)

#### **Data Processing**
- **`build_final.py`**: Consolidates extracted networks into final dataset format
- **`generate_subgraphs.py`**: Creates subgraph representations for analysis
- **`generate_subgraph_text.py`**: Generates natural language descriptions of subgraphs
- **`clean_bn.py`**: Cleans and validates Bayesian Network structures
- **`cpd_calculator.py`**: Computes conditional probability distributions from marginal probabilities
- **`patch_multi_joints.py`**: Applies post-processing patches to joint probability estimates

### Configuration & Data
- **`prism_bn.json`**: Reference Bayesian Network used for evaluation
 
---


### Required Environment Variables

```bash
# Essential (no defaults)
export HF_TOKEN="your_huggingface_token"
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Model-specific (with defaults)
export EXTRACTION_MODEL="extraction-model"       # Default: "extraction-model"
export JUDGE_MODEL="judge-model"                 # Default: "judge-model"
```

 
---

## Data Format

### Input Format (Text Articles)
```json
{
  "id": "article_1",
  "text": "Article content describing causal relationships...",
  "domain": "healthcare"
}
```

### Output Format (Extracted Bayesian Network)
```json
{
  "id": "article_1",
  "nodes": ["node1", "node2", "node3"],
  "edges": [["node1", "node2"], ["node2", "node3"]],
  "states": {
    "node1": ["state_a", "state_b"],
    "node2": ["state_x", "state_y"]
  },
  "cpd": {
    "node1": [[0.6, 0.4]],
    "node2": [[0.7, 0.3], [0.2, 0.8]],
    "node3": [[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
  }
}
```

### Evaluation Results Format
```json
{
  "model": "extraction-model",
  "pass1": {
    "node_f1": 0.85,
    "percent_matched": 0.90,
    "extra_nodes": 2
  },
  "pass2": {
    "state_f1": 0.82,
    "percent_matched": 0.88
  },
  "pass3": {
    "percent_correct_edges": 0.80,
    "percent_spurious": 0.05
  },
  "pass4": {
    "kl_divergence": 0.145
  }
}
```

---

## Configuration Parameters

Key parameters used in evaluation (see scripts for details):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NODE_THRESHOLD` | 0.5 | Similarity threshold for node matching |
| `STATE_THRESHOLD` | 0.5 | Similarity threshold for state matching |
| `LAPLACE_EPS` | 1e-6 | Smoothing for probability distributions |
| `MAX_RETRIES` | 3 | API call retry attempts |
| `DELAY` | 0.3s | Delay between API calls |

---

## Requirements

### System Requirements
- Python 3.8 or higher
- 8GB+ RAM recommended for large datasets
- Network access for LLM API calls

### Python Dependencies
```
anthropic>=0.18.0
huggingface-hub>=0.16.0
networkx>=3.0
wikipedia>=1.4.0
requests>=2.28.0
```

### API Credentials Required
- Hugging Face API token (for judge model inference)
- OpenAI API key (if using GPT models)
- Anthropic API key (if using Claude models)

---

## Usage Examples

### Basic Extraction
```bash
python extractor.py --input articles.json --output networks.jsonl
```

### Comparative Evaluation
```bash
# Evaluate all models
for model in claude deepseek gpt llama qwen gemma; do
    python ${model}_gen_llama_eval.py --input test_data.json
done
```

### Full Pipeline
```bash
# 1. Extract networks
python extractor.py --input raw_articles.json --output extracted.jsonl

# 2. Build final dataset
python build_final.py --input extracted.jsonl --output dataset.json

# 3. Clean and process
python clean_bn.py --input dataset.json --output cleaned.json

# 4. Compute CPDs
python cpd_calculator.py --input cleaned.json --output final_cpds.json

# 5. Generate analysis
python generate_subgraphs.py --input final_cpds.json
python generate_subgraph_text.py --input subgraphs.json
```

---

 

```bash
# Check for API tokens
grep -r "hf_[A-Za-z0-9]" *.py         # Should return no results
grep -r "sk-proj-" *.py               # Should return no results
grep -r "sk_[A-Za-z0-9]" *.py        # Should return no results

# Check for model identifiers
grep -r "DeepSeek\|Claude\|GPT-4\|Qwen" *.py | grep -v "^[[:space:]]*#"
```


---

## Output Files

Each evaluation script generates:
- `anonymized_eval_results/` - Detailed evaluation results per pass
- `anonymized_result_analysis/` - Aggregated analysis and statistics
- CSV files with comparative metrics

---

## Citation

If you use this code in your research, please cite appropriately in your work.

---

## License

This code is provided for research and educational purposes.

---


