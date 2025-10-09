from itertools import product
from collections.abc import Iterable
from typing import Any, Dict, List

def generate_hyperparameter_combinations(param_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given a dict mapping hyperparameter names to iterables of values, return a list of
    dicts representing every combination (cartesian product) of the provided values.
    Example:
        generate_hyperparameter_combinations({
            "lr": [1e-3, 1e-4],
            "batch": [16, 32]
        })
    returns:
        [{"lr": 1e-3, "batch": 16}, {"lr": 1e-3, "batch": 32}, {"lr": 1e-4, "batch": 16}, ...]
    """
    keys = list(param_grid.keys())
    value_lists = []
    for k in keys:
        v = param_grid[k]
        # Treat strings and non-iterables as singletons
        if isinstance(v, str) or not isinstance(v, Iterable):
            value_lists.append([v])
        else:
            value_lists.append(list(v))

    return [dict(zip(keys, combo)) for combo in product(*value_lists)]

