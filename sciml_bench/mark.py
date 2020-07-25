from sciml_bench.benchmarks import BENCHMARKS
from sciml_bench.core.benchmark import MODEL_FUNCS_REGISTRY, DATA_LOADER_REGISTRY, VALIDATION_DATA_LOADER_REGISTRY


def check_benchmark_exists(name):
    benchmark_names = [b.name for b in BENCHMARKS]
    if name not in benchmark_names:
        raise RuntimeError("Could not find match benchmark spec for name {}! \
                Available benchmarks are {}".format(name, [b.name for b in
                    BENCHMARKS]))


def register_model_func(benchmark_name, func):
    MODEL_FUNCS_REGISTRY[benchmark_name].append(func)


def model_function(benchmark_name):
    def decorator(function):
        check_benchmark_exists(benchmark_name)
        MODEL_FUNCS_REGISTRY[benchmark_name].append(function)
        return function

    return decorator


def data_loader(benchmark_name):
    def decorator(class_):
        check_benchmark_exists(benchmark_name)
        DATA_LOADER_REGISTRY[benchmark_name].append(class_)
        return class_

    return decorator


def validation_data_loader(benchmark_name):
    def decorator(class_):
        check_benchmark_exists(benchmark_name)
        VALIDATION_DATA_LOADER_REGISTRY[benchmark_name].append(class_)
        return class_

    return decorator
