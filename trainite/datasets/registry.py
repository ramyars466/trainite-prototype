import json
import importlib.util
import os

DATASET_REGISTRY = {}


def register_dataset(name):
    """
    Decorator to register datasets
    """

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def load_plugin_datasets():
#this reads the json form the plugins folder
    plugins_file = "trainite/plugins/datasets.json"

    if not os.path.exists(plugins_file):
        return

    with open(plugins_file, "r") as f:
        plugins = json.load(f)

    for name, path in plugins.items():
 # Dynamically load the Python file
        if not os.path.exists(path):
            continue

        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "Dataset"):
            DATASET_REGISTRY[name] = module.Dataset
            # If the file has a class called "Dataset", register it

def get_dataset(name):

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not registered")

    return DATASET_REGISTRY[name]