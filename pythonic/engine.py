from typing import Dict, Any, List, Callable
from .schemas import FunctionResults, ExecutionResults
from .util import (
    extract_codeblocks,
    import_functions,
    setup_logger,
)

# Set up logger using the utility function
logger = setup_logger(__name__)


def execute_python_code(
    code: str,
    functions: List[Callable] = [],
    context_variables: Dict[str, Any] = {},
    safe: bool = False,
) -> FunctionResults:
    """
    Execute Python code with given functions and context variables,
    and return the results of function calls and variables defined in the code.

    Args:
        code (str): The Python code to execute.
        functions (List[Callable], optional): A list of functions to make available to the code.
        context_variables (Dict[str, Any], optional): Variables to make available to the code.
        safe (bool, optional): Whether to sandbox the execution environment by restricting dangerous builtins.

    Returns:
        Dict[str, Any]: A dictionary containing the function results, variables defined in the code, and any errors.
    """
    # Define dangerous builtins to restrict
    dangerous_builtins = [
        "exec",
        "eval",
        "execfile",
        "compile",
        "importlib",
        "__import__",
        "input",
    ]

    # Create an execution environment
    env = {"__builtins__": __builtins__}

    # If sandboxing is enabled, restrict dangerous builtins
    if safe:
        env["__builtins__"] = {
            k: v
            for k, v in __builtins__.__dict__.items()
            if k not in dangerous_builtins
        }

    # Record the initial environment keys
    initial_keys = set(env.keys())

    # Add context variables to the execution environment
    if context_variables and isinstance(context_variables, dict):
        env.update(context_variables)

    # A dictionary to store function call results
    call_results = {}

    # Wrap the functions to capture their return values
    def make_wrapper(func_name, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            call_results.setdefault(func_name, []).append(result)
            return result

        return wrapper

    # Add the wrapped functions to the execution environment
    for func in functions:
        env[func.__name__] = make_wrapper(func.__name__, func)

    # Add the typing types to the execution environment
    import_string = "from typing import List, Dict, Any, Union, Tuple, Callable"
    exec(import_string, env)
    exec("import re", env)

    # Execute the code and catch any exceptions
    errors = []
    try:
        exec(code, env)
    except Exception as e:
        errors.append(str(e))

    # Extract variables defined in the code
    variables = {
        k: v
        for k, v in env.items()
        if k not in initial_keys and not k.startswith("__") and not callable(v)
    }

    # Match the call results with the variable names
    for func_name, results in list(call_results.items()):
        for variable_name, variable_value in variables.items():
            for result in results:
                if variable_value == result:
                    call_results[func_name] = variable_name
                    break

    # Return function results, variables, and any errors
    return FunctionResults(results=call_results, data=variables, errors=errors)


def execute_tool_call(
    functions,
    completion,
    show_completion: bool = False,
) -> ExecutionResults:
    errors = []
    results = None

    try:
        if show_completion:
            logger.info(f"Completion: {completion}")

        # Extract code from completion if needed
        code = extract_codeblocks(completion) if "```" in completion else completion

        # Execute the code with mock functions
        results = execute_python_code(code, functions)
        errors.extend(results.errors)

    except Exception as e:
        errors.append(f"Error processing row: {str(e)}")

    return ExecutionResults(
        results=results.results,
        data=results.data,
        errors=errors,
        content=completion,
        is_dry=False,
    )
