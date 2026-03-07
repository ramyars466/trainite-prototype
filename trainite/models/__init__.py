import os
import pkgutil
import importlib
from .transformer import DecoderOnlyTransformer
from .lstm import LSTMModel
from .gru import GRUModel

# Automatically import all model modules
package_dir = os.path.dirname(__file__)

for (_, module_name, _) in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f"{__name__}.{module_name}")