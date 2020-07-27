import importlib
from pathlib import Path
from sciml_bench.core.benchmark import BENCHMARK_REGISTRY

# Register a list of all possible benchmark specifications.
BENCHMARKS = BENCHMARK_REGISTRY


def register_all_objects(module_dir=None):
    from sciml_bench.core.bench_logger import LOGGER

    if module_dir is None:
        module_dir = Path(__file__).parent.absolute()
    else:
        module_dir = Path(module_dir).expanduser()

    LOGGER.debug('Importing modules from {}'.format(module_dir))

    _benchmark_modules = module_dir.glob('**/*.py')
    for module_name in _benchmark_modules:
        module_path = module_name
        module_path = str(module_path)
        LOGGER.debug(module_path)

        try:
            spec = importlib.util.spec_from_file_location(module_path.replace('/', '.'), module_path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
        except ModuleNotFoundError:
            LOGGER.debug('Skipping module {} due to module not found error'.format(module_path))
