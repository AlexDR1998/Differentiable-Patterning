import matplotlib.pyplot as plt
from itertools import product
from collections.abc import Iterable
from typing import Any, Dict, List

def plot_matrix(data):
    plt.figure(figsize=(4,4),dpi=400)
    plt.imshow(data,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    return plt.gca()




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

def generate_hyperparameter_combinations_indexed(param_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    list_of_combinations = generate_hyperparameter_combinations(param_grid)
    list_of_indexed_combinations = []
    for i,hparams in enumerate(list_of_combinations):
        hparams['LIST_INDEX'] = i
        list_of_indexed_combinations.append(hparams)
    return list_of_indexed_combinations