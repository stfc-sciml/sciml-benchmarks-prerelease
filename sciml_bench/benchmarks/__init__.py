from pathlib import Path
from sciml_bench.core.benchmark import BenchmarkSpec


class DMSClassfierSpec(BenchmarkSpec):
    name = 'dms_classifier'
    train_dir = 'train'
    test_dir = 'test'


class EMDenoiseSpec(BenchmarkSpec):
    name = 'em_denoise'
    train_dir = 'train'
    test_dir = 'test'


class SLSTRCloudSpec(BenchmarkSpec):
    name = 'slstr_cloud'
    train_dir = 'pixbox'
    test_dir = 'pixbox'


# Register a list of all possible benchmark specifications.
BENCHMARKS = [
    DMSClassfierSpec,
    EMDenoiseSpec,
    SLSTRCloudSpec
]


def register_all_objects():
    import importlib
    from sciml_bench.core.logging import LOGGER

    module_dir = Path(__file__).parent.parent

    LOGGER.debug('Importing modules from {}'.format(module_dir))

    _benchmark_modules = module_dir.glob('**/*.py')
    for module_name in _benchmark_modules:
        module_path = module_name.relative_to(module_dir.parent)
        module_path = module_path.with_suffix('')
        module_path = str(module_path)
        module_path = module_path.replace('/', '.')
        LOGGER.debug(module_path)
        importlib.import_module(str(module_path))
