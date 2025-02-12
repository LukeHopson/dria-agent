from string import printable

from pydantic import BaseModel
from typing import Dict, Any, List
import json


class FunctionResults(BaseModel):
    """Results from executing functions, including return values, variables and errors."""

    results: Dict[str, Any]
    data: Dict[str, Any]
    errors: List[str]


class ExecutionResults(BaseModel):
    results: Dict[str, Any]
    data: Dict[str, Any]
    errors: List[str]
    content: str
    is_dry: bool

    def dict(self, *args, **kwargs):
        if self.is_dry:
            return {"content": self.content}
        return super().model_dump_json(
            *args, **kwargs, exclude_none=True, exclude_defaults=True
        )

    def __str__(self):
        if self.is_dry:
            return json.dumps({"content": self.content})
        return json.dumps({k: self.data[v] for k, v in self.results.items()})

    def final_answer(self):
        if self.is_dry:
            return "*Dry run has not yet executed anything.*"
        if not self.data or (len(self.data) == 1 and "re" in self.data):
            raise ValueError("No final value")
        return list(self.data.values())[-1]
