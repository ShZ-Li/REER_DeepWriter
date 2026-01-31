# REverse-Engineered Reasoning for Open-Ended Generation
The official code of "REverse-Engineered Reasoning for Open-Ended Generation"

## Release Progress
- [x] Deep Reasoning Synthesis

- [ ] evaluation

- [ ] VeRL based Distributed SFT training

## Synthesis of Deep Reasoning
cd folder: `synthesize_deep_reasoning`

- **Step 0: Update the config.**

  check `config.yaml`:
    ```
      stop_thresh: 0.25 # PPL stopping criterion
      max_step: 10 # max-step stopping criterion
      num_rollouts: 1 # num initial thinking rollouts each query, not tested
      num_expansion: 2 # num expanded node for each segment edits
      file_path: '/path/to/QAcollection.json'
      file_prefix: '/path/to/output/file/folder/'
    ```
  Json format: a list of dicts, where each dict has three keys, `question`, `solution`, `index`

- **Step 1: Start the vLLM server (for vllm_server model type).**
  ```bash 
  export model=/path/to/generator
  export model2=/path/to/basemodel/for/PPL
  bash server.sh
  ```
  We use Qwen2.5-32B-Instruct as the generator, and Qwen3-8B-Base as the model for computing perplexity. We find it faster if we amortize the PPL computation to a smaller model. 

- **Step 2: Run the Deep Reasoning Synthesis with Ray-scheduled multi-workers.**
  ```
  export workdir=${pwd}
  export model=/path/to/generator
  export model2=/path/to/basemodel/for/PPL
  export port=2233
  export rank=0 
  export total=1
  export cname=/path/to/config
  bash synthesis.sh
  ```
  The synthesized trajectories will be dumped to the `file_prefix` path. 

## Using API-based Models (OpenAI, Anthropic, etc.)

Instead of deploying models locally, you can use API-based models like OpenAI or Anthropic for the generator model.

### Configuration

Update your `config.yaml` to use API-based models:

**For OpenAI:**
```yaml
model:
  model_type: "openai"
  model_name: "gpt-4o"  # or any OpenAI model name
  tokenizer_name: "Qwen/Qwen2.5-7B-Instruct"  # Optional: tokenizer for prompt formatting
  model_args:
    max_tokens: 8000
    top_p: 0.85
    temperature_range: [0.8, 0.8]
    api_key: "sk-your-api-key"  # Optional if OPENAI_API_KEY env var is set
    base_url: "https://api.openai.com/v1"  # Optional, customize for OpenAI-compatible APIs
  prompt_type: "tokenizer"
```

**For Anthropic:**
```yaml
model:
  model_type: "anthropic"
  model_name: "claude-3-5-sonnet-20241022"  # or any Anthropic model name
  tokenizer_name: "Qwen/Qwen2.5-7B-Instruct"  # Optional: tokenizer for prompt formatting
  model_args:
    max_tokens: 8000
    top_p: 0.85
    temperature_range: [0.8, 0.8]
    api_key: "sk-ant-your-api-key"  # Optional if ANTHROPIC_API_KEY env var is set
    base_url: "https://api.anthropic.com"  # Optional
  prompt_type: "tokenizer"
```

### Running with API Models

When using API-based models, you don't need to specify `--port` or `--model` arguments:

```bash
export workdir=${pwd}
export rank=0 
export total=1
export cname=/path/to/config_with_api_model.yaml

# For API models, --port and --model are optional
python synthesize.py --config_file $cname --rank $rank --total-ranks $total
```

### Environment Variables

You can set API keys and base URLs via environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-api-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"  # Optional
```

### Supported Model Types

- `hf` - HuggingFace local models
- `vllm` - vLLM local inference
- `vllm_server` - vLLM server (default)
- `openai` - OpenAI API (also supports OpenAI-compatible APIs)
- `anthropic` - Anthropic API

### Notes

1. When using API-based models, the `port` parameter in `model_args` is not required.
2. API-based models do not provide token-level log probabilities, so the perplexity-based refinement features will have limited functionality.
3. The `base_url` parameter allows you to use OpenAI-compatible APIs (e.g., Azure OpenAI, local API servers). 
