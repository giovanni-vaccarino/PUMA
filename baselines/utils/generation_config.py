"""
Load generation parameters (temperature, top_p, top_k) from a model's
generation_config.json, matching the PUMA pipeline's approach.
"""

from transformers import GenerationConfig


def load_generation_params(model_name: str) -> dict:
    """Load temperature/top_p/top_k from the model's generation_config.json.

    Returns a dict with keys: temperature, top_p, top_k.
    Falls back to conservative defaults if loading fails.
    """
    try:
        gc = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
        temperature = getattr(gc, "temperature", 0.6)
        top_p = getattr(gc, "top_p", 0.95)
        top_k = getattr(gc, "top_k", -1)  # -1 = disabled in vLLM
    except Exception as e:
        print(f"[generation_config] Failed to load from {model_name}: {e}")
        print("[generation_config] Using defaults: temperature=0.6, top_p=0.95, top_k=-1")
        temperature = 0.6
        top_p = 0.95
        top_k = -1

    params = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    print(f"[generation_config] {model_name.split('/')[-1]}: "
          f"temperature={temperature}, top_p={top_p}, top_k={top_k}")
    return params
