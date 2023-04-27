_model_type = []
_dataset_type = {}
_model_entrypoints = {}  # mapping of model names to entrypoint fns


def register_model(cls):
    _model_type.append(cls.__name__)
    return cls


def get_model_type():
    return _model_type


def get_models(model_name):
    return _model_entrypoints[model_name]


def register(fn):
    model_name = fn.__name__
    _model_entrypoints[model_name] = fn
    return fn


def register_dataset(cls):
    dataset_name = cls.__name__
    _dataset_type[dataset_name] = cls
    return cls


def get_dataset_type():
    return _dataset_type.keys()


def get_dataset(dataset_name):
    return _dataset_type[dataset_name]
