import asyncio
from typing import Dict, Any, List, Callable

from .schemas import FunctionResults, ExecutionResults
from .util import (
    extract_codeblocks,
    setup_logger,
)

# Set up logger using the utility function
logger = setup_logger(__name__)


def _create_execution_env(safe: bool = False) -> Dict[str, Any]:
    """Create and return a sandboxed execution environment if safe=True."""
    dangerous_builtins = [
        "exec",
        "eval",
        "execfile",
        "compile",
        "importlib",
        "__import__",
        "input",
    ]

    env = {"__builtins__": __builtins__}

    if safe:
        env["__builtins__"] = {
            k: v
            for k, v in __builtins__.__dict__.items()
            if k not in dangerous_builtins
        }

    return env


def _setup_env_imports(env: Dict[str, Any]) -> None:
    """Set up common imports in the execution environment."""
    exec("from typing import List, Dict, Any, Union, Tuple, Callable", env)
    exec("import re", env)
    exec("from datetime import datetime, timedelta", env)


def _make_sync_wrapper(
    func_name: str, func: Callable, call_results: Dict, errors: List
) -> Callable:
    """Create a synchronous wrapper function that captures return values."""

    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            call_results.setdefault(func_name, []).append(result)
            return result
        except Exception as e:
            errors.append(f"Error in {func_name}: {str(e)}")
            raise

    return wrapper


async def _make_async_wrapper(
    func_name: str, func: Callable, call_results: Dict, errors: List
) -> Callable:
    """Create an async wrapper function that captures return values."""
    if asyncio.iscoroutinefunction(func):

        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                call_results.setdefault(func_name, []).append(result)
                return result
            except Exception as e:
                errors.append(f"Error in {func_name}: {str(e)}")
                raise

    else:

        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                call_results.setdefault(func_name, []).append(result)
                return result
            except Exception as e:
                errors.append(f"Error in {func_name}: {str(e)}")
                raise

    return wrapper


def _match_results_to_variables(call_results: Dict, variables: Dict) -> None:
    """Match function call results with variable names."""
    for func_name, results in list(call_results.items()):
        for variable_name, variable_value in variables.items():
            for result in results:
                if variable_value == result:
                    call_results[func_name] = variable_name
                    break


def execute_python_code(
    code: str,
    functions: List[Callable] = [],
    context_variables: Dict[str, Any] = {},
    safe: bool = False,
) -> FunctionResults:
    """
    Execute Python code with given functions and context variables.

    Args:
        code: The Python code to execute.
        functions: List of functions to make available to the code.
        context_variables: Variables to make available to the code.
        safe: Whether to sandbox the execution environment.

    Returns:
        FunctionResults containing results, variables and any errors.
    """
    env = _create_execution_env(safe)
    initial_keys = set(env.keys())

    if context_variables and isinstance(context_variables, dict):
        env.update(context_variables)

    call_results = {}
    errors = []

    for func in functions:
        env[func.__name__] = _make_sync_wrapper(
            func.__name__, func, call_results, errors
        )

    _setup_env_imports(env)

    try:
        exec(code, env)
    except Exception as e:
        errors.append(str(e))

    variables = {
        k: v
        for k, v in env.items()
        if k not in initial_keys and not k.startswith("__") and not callable(v)
    }

    _match_results_to_variables(call_results, variables)

    return FunctionResults(results=call_results, data=variables, errors=errors)


async def async_execute_python_code(
    code: str,
    functions: List[Callable] = [],
    context_variables: Dict[str, Any] = {},
    safe: bool = False,
) -> FunctionResults:
    """
    Asynchronously execute Python code with given functions and context variables.

    Args:
        code: The Python code to execute.
        functions: List of functions to make available to the code.
        context_variables: Variables to make available to the code.
        safe: Whether to sandbox the execution environment.

    Returns:
        FunctionResults containing results, variables and any errors.
    """
    env = _create_execution_env(safe)
    initial_keys = set(env.keys())

    if context_variables and isinstance(context_variables, dict):
        env.update(context_variables)

    call_results = {}
    errors = []
    env["asyncio"] = asyncio

    for func in functions:
        wrapper = await _make_async_wrapper(func.__name__, func, call_results, errors)
        env[func.__name__] = wrapper
        if asyncio.iscoroutinefunction(func):
            code = code.replace(func.__name__, f"await {func.__name__}")

    _setup_env_imports(env)

    try:
        async_code = "async def __async_exec():\n"
        async_code += "".join(f"    {line}\n" for line in code.splitlines())
        async_code += "\n    return locals()"

        exec_globals = {}
        exec(async_code, env, exec_globals)
        result = await exec_globals["__async_exec"]()
        env.update(result)
    except Exception as e:
        errors.append(str(e))

    variables = {
        k: v
        for k, v in env.items()
        if k not in initial_keys and not k.startswith("__") and not callable(v)
    }

    _match_results_to_variables(call_results, variables)

    return FunctionResults(results=call_results, data=variables, errors=errors)


def execute_tool_call(
    functions: List[Callable],
    completion: str,
    show_completion: bool = False,
) -> ExecutionResults:
    """
    Execute a tool call with the given functions and completion.

    Args:
        functions: List of functions to make available.
        completion: Code completion to execute.
        show_completion: Whether to log the completion.

    Returns:
        ExecutionResults containing the execution outcome.
    """
    errors = []
    results = None

    try:
        if show_completion:
            logger.info(f"Completion: {completion}")

        code = extract_codeblocks(completion) if "```" in completion else completion
        results = execute_python_code(code, functions)
        errors.extend(results.errors)

    except Exception as e:
        errors.append(f"Error processing row: {str(e)}")

    return ExecutionResults(
        results=results.results if results else None,
        data=results.data if results else None,
        errors=errors,
        content=completion,
        is_dry=False,
    )


async def async_execute_tool_call(
    functions: List[Callable],
    completion: str,
    show_completion: bool = False,
) -> ExecutionResults:
    """
    Asynchronously execute a tool call with the given functions and completion.

    Args:
        functions: List of functions to make available.
        completion: Code completion to execute.
        show_completion: Whether to log the completion.

    Returns:
        ExecutionResults containing the execution outcome.
    """
    errors = []
    results = None

    try:
        if show_completion:
            logger.info(f"Completion: {completion}")

        code = extract_codeblocks(completion) if "```" in completion else completion
        results = await async_execute_python_code(code, functions)
        errors.extend(results.errors)

    except Exception as e:
        errors.append(f"Error processing row: {str(e)}")

    return ExecutionResults(
        results=results.results if results else None,
        data=results.data if results else None,
        errors=errors,
        content=completion,
        is_dry=False,
    )
