import torch
from typing import Any, Optional

def count_params(model: torch.nn.Module, verbose: bool = False) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count parameters for.
        verbose (bool): If True, print the parameter count. Defaults to False.

    Returns:
        int: The total number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Total parameters: {total_params}")
    return total_params

def exists(x: Any) -> bool:
    """
    Check if a variable exists and is not None.

    Args:
        x (Any): The variable to check.

    Returns:
        bool: True if the variable exists and is not None, False otherwise.
    """
    return x is not None

def default(val: Any, d: Any) -> Any:
    """
    Return a default value if the given value doesn't exist.

    Args:
        val (Any): The value to check.
        d (Any): The default value to return if val doesn't exist.

    Returns:
        Any: val if it exists, otherwise d.
    """
    return val if exists(val) else d

def instantiate_from_config(config: dict) -> Any:
    """
    Instantiate an object from a configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration for the object.
                       It should have a 'target' key specifying the object's class path,
                       and optionally a 'params' key for initialization parameters.

    Returns:
        Any: An instance of the specified class.

    Raises:
        ImportError: If the specified class cannot be imported.
        TypeError: If the specified class is not callable.
    """
    if not "target" in config:
        raise ValueError("Config must contain a 'target' key specifying the object to instantiate.")

    module_name, class_name = config["target"].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    if not callable(cls):
        raise TypeError(f"The specified target '{config['target']}' is not callable.")

    params = config.get("params", {})
    return cls(**params)
