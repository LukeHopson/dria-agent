# tiny-agent-α

Tiny-Agent-α is an extension of [Dria-Agent-a](https://huggingface.co/collections/driaforall/dria-agent-a-67a61f4b7d3d544fe5d3cd8a=, trained on top of the Qwen2.5-Coder series to be used in edge devices. 
These models are carefully fine-tuned with quantization aware training to minimize performance degradation after quantization. 
The smallest model is 0.5B with 4bit quantization (398MB on disk), and the largest model is 3B with 4bit quantization.

### Installation

To install the package run: (Best for CPU inference)
```bash
pip install dria_agent
```
To use MLX as backend for macOS. 
```bash
pip install 'dria_agent[mlx]'
```
For HuggingFace backend for GPU.
```bash
pip install 'dria_agent[huggingface]'
```

In order to use extra tooling select a backend (e.g. mlx) and run:
```bash
pip install 'dria_agent[mlx, tools]'
```

### Models

| Model                  | Description                                | HF Download Link                                                                                                         | Ollama Tag                         | Size   |
|------------------------|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------|--------|
| Tiny-Agent-a-3B (8bit) | High performance and reasoning             | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-3B/resolve/main/dria-agent-a-3b.Q8_0.gguf?download=true)       | driaforall/tiny-agent-a:3B-q8_0  | 3.3 GB |
| Tiny-Agent-a-3B (4bit) | Tradeoff 3B quality for memory             | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-3B/resolve/main/dria-agent-a-3b.Q4_K_M.gguf?download=true)     | driaforall/tiny-agent-a:3B-q4_K_M | 1.9 GB |
| Tiny-Agent-a-1.5B (8bit) | Balanced performance and speed             | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q8_0.gguf?download=true)   | driaforall/tiny-agent-a:1.5B-q8_0 | 1.6 GB |
| Tiny-Agent-a-1.5B (4bit) | Faster CPU inference, performance tradeoff | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q8_0.gguf?download=true)   | driaforall/tiny-agent-a:1.5B-q4_K_M | 986 MB |
| Tiny-Agent-a-0.5B (8bit) | Ultra-light                                | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q4_K_M.gguf?download=true) | driaforall/tiny-agent-a:0.5B-q8_0 | 531 MB |

a fast and powerful tool calling model designed to run on edge devices.

3B 8bit high performance and quality [download](https://huggingface.co/driaforall/Tiny-Agent-a-3B/resolve/main/dria-agent-a-3b.Q8_0.gguf?download=true)

3B 4bit balanced performance and speed [download](https://huggingface.co/driaforall/Tiny-Agent-a-3B/resolve/main/dria-agent-a-3b.Q4_K_M.gguf?download=true)

1.5B 8bit fast CPU inference [download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q8_0.gguf?download=true)

1.5B 4bit faster CPU inference [download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q8_0.gguf?download=true)

0.5B 8bit ultra-light [download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q4_K_M.gguf?download=true)


