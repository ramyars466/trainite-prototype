import importlib.util
import sys
import os


def load_dataset_plugin(file_path):
    """
    Dynamically load a dataset plugin from a python file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    module_name = os.path.basename(file_path).replace(".py", "")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module



