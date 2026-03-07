DATASET_REGISTRY = {}

def register_dataset(name):

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(name):

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found")

    return DATASET_REGISTRY[name]