import os
import pkgutil
import importlib
from .registry import register_dataset, get_dataset

# import built-in datasets
from .string_reverse import StringReversalDataset

# Automatically import all dataset modules
package_dir = os.path.dirname(__file__)

for (_, module_name, _) in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f"{__name__}.{module_name}")