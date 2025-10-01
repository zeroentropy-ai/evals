import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Cache the LLM instance to avoid reloading
_llm_cache = {}

def rerank_vllm(
    query: str,
    documents: list[str],
    model_path: str,
    use_fp8: bool = False,
) -> list[float]:
    """Rerank documents using vLLM (with optional FP8 quantization)."""
    
    cache_key = f"{model_path}_fp8={use_fp8}"
    
    # Cache the LLM to avoid reloading
    if cache_key not in _llm_cache:
        print(f"Loading model from {model_path} (FP8={use_fp8})...")
        llm_kwargs = {
            "model": model_path,
            "max_model_len": 2048,
        }
        if use_fp8:
            llm_kwargs["quantization"] = "fp8"
            llm_kwargs["gpu_memory_utilization"] = 0.9
        
        _llm_cache[cache_key] = {
            "llm": LLM(**llm_kwargs),
            "tokenizer": AutoTokenizer.from_pretrained(model_path),
        }
    
    llm = _llm_cache[cache_key]["llm"]
    tokenizer = _llm_cache[cache_key]["tokenizer"]
    
    # Get the "Yes" token ID
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    
    # Prepare texts for each document
    texts = []
    for doc in documents:
        messages = [
            {"role": "system", "content": query},
            {"role": "user", "content": doc},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
        )
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=0,
        logprobs=1,
        repetition_penalty=1,
        allowed_token_ids=[yes_token_id],
    )
    
    outputs = llm.generate(texts, sampling_params)
    
    # Extract scores
    scores = []
    for output in outputs:
        if output.outputs and output.outputs[0].logprobs:
            first_token_logprobs = output.outputs[0].logprobs[0]
            
            if yes_token_id in first_token_logprobs:
                yes_logprob = first_token_logprobs[yes_token_id].logprob
            else:
                yes_logprob = -100.0
            
            score = 1 / (1 + math.exp(-(yes_logprob / 5.0)))
            scores.append(score)
        else:
            scores.append(0.0)
    
    return scores
