LM_STUDIO_URL = "http://localhost:1234/v1"
OLLAMA_URL = "http://localhost:11434/v1"
VLLM_URL = "http://localhost:8000/v1"

PROVIDER_URLS = {
    "lm_studio": (LM_STUDIO_URL, "api_key"),
    "ollama": (OLLAMA_URL, "api_key"),
    "vllm": (VLLM_URL, "api_key"),
}
