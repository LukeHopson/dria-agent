# Dria Agent

A package to use the fastest and most performant tool calling agents on edge devices, `tiny-agent-α`.

Tiny-Agent-α is an extension of [Dria-Agent-a](https://huggingface.co/collections/driaforall/dria-agent-a-67a61f4b7d3d544fe5d3cd8a=), trained on top of the Qwen2.5-Coder series to be used in edge devices. 
These models are carefully fine-tuned with quantization aware training to minimize performance degradation after quantization. 
The smallest model is 0.5B with 4bit quantization (398MB on disk), and the largest model is 3B with 4bit quantization.


https://github.com/user-attachments/assets/c50b71f9-5d6e-4f67-8650-1ffbc2000575


### Features

**One-shot Parallel Multiple Function Calls**

The model can can utilise many synchronous processes in one chat turn to arrive to a solution, which would require other function calling models multiple turns of conversation.

**Free-form Reasoning and Actions**

The model provides reasoning traces freely in natural language and the actions in between ```python ``` blocks, as it already tends to do without special prompting or tuning. 

**On-the-fly Complex Solution Generation**

The solution provided by the model is essentially a Python program with the exclusion of some "risky" builtins like exec, eval and compile (see full list in Quickstart below). 



https://github.com/user-attachments/assets/5f7cbd26-7ba3-46aa-926f-4ac68de5ccb0



#### Edge Device Optimized:
- Supports mlx, ollama, and transformers (Hugging Face).
- Includes built-in support for macOS, Gmail, search, and more.
- Uses similarity search to efficiently select relevant tools.
- Optimized for Edge


### Installation

To install the package run:
```bash
pip install dria_agent # Best for CPU inference
pip install 'dria_agent[mlx]' # To use MLX as backend for macOS. 
pip install 'dria_agent[huggingface]' # HuggingFace/transformers backend for GPU.
pip install 'dria_agent[tools]' # In order to use factory tools in package
```

### Quick Start

Write your functions in pure python, decorate them with @tool to expose them to the agent.

````python
from dria_agent import tool

@tool
def check_availability(day: str, start_time: str, end_time: str) -> bool:
    """
    Checks if a given time slot is available.

    :param day: The date in "YYYY-MM-DD" format.
    :param start_time: The start time of the desired slot (HH:MM format, 24-hour).
    :param end_time: The end time of the desired slot (HH:MM format, 24-hour).
    :return: True if the slot is available, otherwise False.
    """
    # Mock implementation
    if start_time == "12:00" and end_time == "13:00":
        return False
    return True

````

Create an agent:

```python
from dria_agent import ToolCallingAgent

agent = ToolCallingAgent(
    tools=[check_availability]
)

```

#### Running Queries

Use agent.run(query) to execute tasks with tools.

```python
execution = agent.run("Check my calendar for tomorrow noon", print_results=True)
```

Tiny-Agent-3B outputs:

````
let me help you check your availability for a 1-hour meditation session       
starting at noon tomorrow.                                                    
                                                                                
Step-by-step reasoning:                                                       
 1. We need to check availability for a specific time slot (noon)              
 2. The duration is 1 hour, so we'll use the same start and end times          
 3. Since it's tomorrow, we should format the date as "YYYY-MM-DD"             
 4. Use the check_availability() function with these parameters                
                                                                                
Here's the code to check your availability:                                   
                                                                                
```python                                                                     
tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")          
start_time = "12:00"  # Noon in 24-hour format                                
end_time = "13:00"   # One hour after noon                                    
                                                                                
availability = check_availability(tomorrow, start_time, end_time)             
```                                                                           
                                                                                
The code will:                                                                
- Calculate tomorrow's date using datetime and timedelta                      
- Set the time slot to noon (12:00) for 1 hour duration                       
- Check if this time slot is available using the check_availability function  
                                                                                
The availability variable will contain True if you're available, or False if  
not.

````

#### Modes

**run**

- **query (str)**: The user query to process.
- **dry_run (bool, default=False)**: If True, only performs inference—no tool execution.
- **show_completion (bool, default=True)**: Displays the model’s raw output before tool execution.
- **num_tools (int, default=2)**: Selects the best K tools for inference (using similarity search).
  - *Allows handling thousands of tools efficiently*.
  - * perform best with 4-5 tools max*.
- **print_results (bool, default=True)**: Prints execution results.

**run_feedback()**

**CLI**




## Models

A fast and powerful tool calling model designed to run on edge devices.


| Model                  | Description                                | HF Download Link                                                                                                         | Ollama Tag                         | Size   |
|------------------------|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------|--------|
| Tiny-Agent-a-3B (8bit) | High performance and reasoning             | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-3B/resolve/main/dria-agent-a-3b.Q8_0.gguf?download=true)       | driaforall/tiny-agent-a:3B-q8_0  | 3.3 GB |
| Tiny-Agent-a-3B (4bit) | Tradeoff 3B quality for memory             | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-3B/resolve/main/dria-agent-a-3b.Q4_K_M.gguf?download=true)     | driaforall/tiny-agent-a:3B-q4_K_M | 1.9 GB |
| Tiny-Agent-a-1.5B (8bit) | Balanced performance and speed             | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q8_0.gguf?download=true)   | driaforall/tiny-agent-a:1.5B-q8_0 | 1.6 GB |
| Tiny-Agent-a-1.5B (4bit) | Faster CPU inference, performance tradeoff | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q8_0.gguf?download=true)   | driaforall/tiny-agent-a:1.5B-q4_K_M | 986 MB |
| Tiny-Agent-a-0.5B (8bit) | Ultra-light                                | [Download](https://huggingface.co/driaforall/Tiny-Agent-a-1.5B/resolve/main/dria-agent-a-1.5b.Q4_K_M.gguf?download=true) | driaforall/tiny-agent-a:0.5B-q8_0 | 531 MB |



## Evaluation & Performance

We evaluate the model on the **Dria-Pythonic-Agent-Benchmark ([DPAB](https://github.com/firstbatchxyz/function-calling-eval)):** The benchmark we curated with a synthetic data generation +model-based validation + filtering and manual selection to evaluate LLMs on their Pythonic function calling ability, spanning multiple scenarios and tasks. See [blog](https://huggingface.co/blog/andthattoo/dpab-a) for more information.

Below are the DPAB results: 

Current benchmark results for various models **(strict)**:

| Model Name                      | Pythonic | JSON |
|---------------------------------|----------|------|
| **Closed Models**               |          |      |
| Claude 3.5 Sonnet              | 87       | 45   |
| gpt-4o-2024-11-20              | 60       | 30   |
| **Open Models**                 |          |      |
| **> 100B Parameters**           |          |      |
| DeepSeek V3 (685B)             | 63       | 33   |
| MiniMax-01                     | 62       | 40   |
| Llama-3.1-405B-Instruct        | 60       | 38   |
| **> 30B Parameters**            |          |      |
| Qwen-2.5-Coder-32b-Instruct    | 68       | 32   |
| Qwen-2.5-72b-instruct          | 65       | 39   |
| Llama-3.3-70b-Instruct         | 59       | 40   |
| QwQ-32b-Preview                | 47       | 21   |
| **< 20B Parameters**           |          |      |
| Phi-4 (14B)                    | 55       | 35   |
| Qwen2.5-Coder-7B-Instruct      | 44       | 39   |
| Qwen-2.5-7B-Instruct           | 47       | 34   |
| **Tiny-Agent-a-3B**               | **72**       | 34   |
| Qwen2.5-Coder-3B-Instruct      | 26       | 37   |
| **Tiny-Agent-a-1.5B**               | **73**       | 30   |


#### Citation

```
@misc{Dria-Agent-a,
      url={https://huggingface.co/blog/andthattoo/dria-agent-a},
      title={Dria-Agent-a},
      author={"andthattoo", "Atakan Tekparmak"}
}
```

