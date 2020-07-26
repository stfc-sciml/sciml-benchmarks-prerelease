from sciml_bench.core.benchmark import BENCHMARK_REGISTRY, MODEL_FUNCS_REGISTRY, DATA_LOADER_REGISTRY, VALIDATION_DATA_LOADER_REGISTRY


def benchmark_spec(class_):
    BENCHMARK_REGISTRY[class_.name] = class_
    return class_


def model_function(benchmark_name):
    def decorator(function):
        MODEL_FUNCS_REGISTRY[benchmark_name].append(function)
        return function

    return decorator


def data_loader(benchmark_name):
    def decorator(class_):
        DATA_LOADER_REGISTRY[benchmark_name].append(class_)
        return class_

    return decorator


def validation_data_loader(benchmark_name):
    def decorator(class_):
        VALIDATION_DATA_LOADER_REGISTRY[benchmark_name].append(class_)
        return class_

    return decorator
