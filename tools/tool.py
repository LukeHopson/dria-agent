import inspect
import logging

logger = logging.getLogger(__name__)


class ToolCall:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.docstring = func.__doc__ or ""
        if self.docstring == "":
            logger.info(
                "No docstring provided, please add detailed docstrings for effective tool calling."
            )
        self.signature = inspect.signature(func)

        # Extract parameters information.
        self.params = {}
        for param_name, param in self.signature.parameters.items():
            self.params[param_name] = {
                "annotation": (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else None
                ),
                "default": (
                    param.default if param.default != inspect.Parameter.empty else None
                ),
                "kind": param.kind,
            }

        # Extract return type.
        self.return_type = (
            self.signature.return_annotation
            if self.signature.return_annotation != inspect.Signature.empty
            else None
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        # Build the parameter list string.
        params_list = []
        for name, meta in self.params.items():
            if meta["annotation"] is not None:
                # If the annotation is a type, use its __name__, otherwise convert it to string.
                if isinstance(meta["annotation"], type):
                    annotation_str = meta["annotation"].__name__
                else:
                    annotation_str = str(meta["annotation"])
                params_list.append(f"{name}: {annotation_str}")
            else:
                params_list.append(name)
        params_str = ", ".join(params_list)

        # Build the function signature line.
        if self.return_type is not None:
            if isinstance(self.return_type, type):
                ret_type_str = self.return_type.__name__
            else:
                ret_type_str = str(self.return_type)
            sig_line = f"def {self.name}({params_str}) -> {ret_type_str}:"
        else:
            sig_line = f"def {self.name}({params_str}):"

        # Build the docstring block with proper indentation.
        doc_lines = self.docstring.strip().splitlines()
        doc_block = '    """\n'
        for line in doc_lines:
            doc_block += f"    {line}\n"
        doc_block += '    """'

        # Combine signature, docstring, and a placeholder 'pass'.
        return f"{sig_line}\n{doc_block}\n    pass"


def tool(func):
    """
    Decorator that converts a function into a ToolCall instance,
    extracting its parameters, return type, and docstring.
    """
    return ToolCall(func)


if __name__ == "__main__":

    @tool
    def find_next_available_slot(user_id: str, event_duration: int) -> str:
        """
        Finds the next available time slot in the user's calendar for rescheduling a meeting.

        :param user_id: The unique identifier of the user (e.g., Samantha Reynolds).
        :param event_duration: Duration of the event in minutes.
        :return: A string representing the next available time slot in the format "HH:MM".
        :raises ValueError: If user_id is invalid or no suitable time slot is found.
        """
        return "09:00"

    print(find_next_available_slot)

    @tool
    def sum(x, y) -> int:
        return x + y

    print(sum)
