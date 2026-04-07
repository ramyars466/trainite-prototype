import json
import importlib.util
import os

MODEL_REGISTRY = {}


def register_model(name):
    """
    Decorator for registering models
    """

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def load_plugin_models():

    plugins_file = "trainite/plugins/models.json"

    if not os.path.exists(plugins_file):
        return

    with open(plugins_file, "r") as f:
        plugins = json.load(f)

    for name, path in plugins.items():

        if not os.path.exists(path):
            continue

        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module) # Actually runs the file!

        if hasattr(module, "Model"):
            MODEL_REGISTRY[name] = module.Model# Now it's in the registry!


def get_model(name):

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not registered")

    return MODEL_REGISTRY[name]