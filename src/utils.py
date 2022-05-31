from functools import update_wrapper
from typing import Callable, Any


class CountCalls:

    def __init__(self, func: Callable) -> None:
        update_wrapper(self, func)
        self.func = func
        self.n_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.n_calls += 1
        return self.func(*args, **kwargs)
