import os


# Optimization flags
os.environ['CUDA_CACHE_DISABLE'] = '0'

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'

os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

import string
import logging
import warnings
import sys
import yaml
import click
import click_config_file
from datetime import datetime
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import horovod.tensorflow as hvd

import sciml_bench
from sciml_bench.core.report import create_report
from sciml_bench.core.logging import LOGGER
from sciml_bench.core.download import download_datasets

def yaml_provider(file_path, cmd_name):
    with open(file_path) as config_data:
        cfg = yaml.load(config_data, yaml.SafeLoader)
        return cfg

def print_header():
    if hvd.rank() == 0:
        text = """

               _           _   _                     _
              (_)         | | | |                   | |
      ___  ___ _ _ __ ___ | | | |__   ___ _ __   ___| |__
     / __|/ __| | '_ ` _ \| | | '_ \ / _ \ '_ \ / __| '_ \\
     \__ \ (__| | | | | | | | | |_) |  __/ | | | (__| | | |
     |___/\___|_|_| |_| |_|_| |_.__/ \___|_| |_|\___|_| |_|



        """
        sys.stdout.write(text)
        sys.stdout.write("\n\n")

    LOGGER.info('Version: %s', sciml_bench.__version__)

    from mpi4py import MPI
    data = (MPI.Get_processor_name(), hvd.local_size())
    _comm = MPI.COMM_WORLD
    data = _comm.bcast(data, root=0)

    data = [data] if not isinstance(data, list) else data

    plurality = 'es' if len(data) > 1 else ''
    for node_name, local_size in data:
        LOGGER.info('%s has %s process%s', node_name, local_size, plurality)

def set_environment_variables(cpu_only=False, use_amp=False, **kwargs):
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    LOGGER.setLevel(kwargs['log_level'])
    if kwargs['log_level'] < logging.INFO:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'


@click.group()
@click.pass_context
def cli(ctx, tracking_uri=None, **kwargs):
    pass

@cli.command('list', help='List benchmarks')
@click.argument('name', default='all', type=click.Choice(['all', 'benchmarks', 'datasets']))
@click.pass_context
def cmd_list(ctx, name, **kwargs):
    if name == 'benchmarks' or name == 'all':
        click.echo('Benchmarks\n')

        for benchmark in BENCHMARKS:
            click.echo(benchmark.name)

    if name == 'datasets' or name == 'all':
        click.echo('')
        click.echo('Datasets\n')

        for benchmark in BENCHMARKS:
            path = Path(ctx.obj['data_dir']).joinpath(benchmark.name.replace('-', '_'))
            downloaded = path.exists()
            click.echo('{}\t\tDownloaded: {}'.format(benchmark.name, downloaded))


@cli.command(help='Run the DMS Classifier Benchmark', hidden=True)
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.option('--epochs', default=10, help='Set number of epochs')
@click.option('--loss', default='binary_crossentropy', help='Set loss function to use')
@click.option('--batch-size', default=256, help='Set the batch size for training & test')
@click.option('--learning-rate', default=1e-4, help='Set the learning rate')
@click.option('--metrics', '-m', default=['accuracy'], multiple=True, help='Set the metrics to output')
def dms_classifier(ctx, **kwargs):
    import sciml_bench.dms_classifier.main as dms_classifier_mod
    _run_benchmark(dms_classifier_mod, ctx, **kwargs)

@cli.command(help='Run the Electron Microscopy Denoise Benchmark', hidden=True)
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.option('--epochs', default=10, help='Set number of epochs')
@click.option('--loss', default='mse', help='Set loss function to use')
@click.option('--batch-size', default=256, help='Set the batch size for training & test')
@click.option('--learning-rate', default=0.01, help='Set the learning rate')
@click.option('--metrics', '-m', default=[], multiple=True, help='Set the metrics to output')
def em_denoise(ctx, **kwargs):
    import sciml_bench.em_denoise.main as em_denoise_mod
    _run_benchmark(em_denoise_mod, ctx, **kwargs)

@cli.command(help='Run the SLSTR Cloud Segmentation Benchmark', hidden=True)
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.option('--epochs', default=30, help='Set number of epochs')
@click.option('--loss', default='binary_crossentropy', help='Set loss function to use')
@click.option('--batch-size', default=6, help='Set the batch size for training & test')
@click.option('--learning-rate', default=0.001, help='Set the learning rate')
@click.option('--metrics', '-m', default=['accuracy'], multiple=True, help='Set the metrics to output')
def slstr_cloud(ctx, **kwargs):
    import sciml_bench.slstr_cloud.main as slstr_cloud_mod
    _run_benchmark(slstr_cloud_mod, ctx, **kwargs)

# List of all benchmark entrypoint functions
BENCHMARKS = [
    dms_classifier,
    em_denoise,
    slstr_cloud
]

# Dict of all benchmark names -> benchmark functions
BENCHMARK_DICT = {b.name: b for b in BENCHMARKS}

@cli.command(help='Run SciML benchmarks')
@click.argument('benchmark_names', nargs=-1, type=click.Choice(['all', ] + [b.name for b in BENCHMARKS]))
@click.option('--data-dir', default='data', help='Data directory location', envvar='SCIML_BENCH_DATA_DIR')
@click.option('--model-dir', default='sciml-bench-out', type=str, help='Output directory for model results', envvar='SCIML_BENCH_MODEL_DIR')
@click.option('--lr-warmup', default=3, type=int, help='Number of epochs over which to scale the learning rate.')
@click.option('--cpu-only', default=False, is_flag=True, help='Disable GPU execution')
@click.option('--use-amp', default=False, is_flag=True, help='Enable Automatic Mixed Precision')
@click.option('--exec-mode', default='train_and_predict', type=click.Choice(['train', 'train_and_predict', 'predict']), help='Set the execution mode')
@click.option('--log-batch', default=False, is_flag=True, help='Whether to log metrics by batch or by epoch')
@click.option('--log-interval', default=0.5, help='Logging interval for system metrics')
@click.option('--seed', default=42, type=int, help='Random seed to use for initialization')
@click.option('--verbosity', default=2, type=int, help='Verbosity level to use. 0 is silence, 3 is maximum information')
@click.option('--log-level', default=logging.INFO, type=int, help='Log level to use for printing to stdout')
@click.option('--skip', default=True, type=bool, help='Whether to skip or exit on encountering an exception')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
def run(ctx, benchmark_names, skip=True, **params):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    with warnings.catch_warnings():
        hvd.init()
        ctx.ensure_object(dict)
        ctx.obj.update(params)

    set_environment_variables(**params)

    LOGGER.setLevel(logging.ERROR)

    if params.get('verbosity') > 2:
        print_header()
        LOGGER.setLevel(logging.INFO)

    model_dir = ctx.obj['model_dir']
    data_dir = ctx.obj['data_dir']

    data_dir = Path(data_dir)

    if not data_dir.exists():
        click.echo("Data directory {} does not exist!".format(data_dir))
        click.abort()

    LOGGER.info('Model directory is: %s', str(model_dir))
    LOGGER.info('Data directory is: %s', str(data_dir))

    # If no benchmarks specified or all then run everything
    if len(benchmark_names) == 0 or 'all' in benchmark_names:
        benchmark_names = BENCHMARK_DICT.keys()

    # Sanity check: do all the benchmarks exist?
    for name in benchmark_names:
        if name not in BENCHMARK_DICT:
            LOGGER.error('Benchmark {} does not exist!'.format(name))
            click.abort()

    # Log which benchmarks we will run
    LOGGER.info('Selected the following benchmarks:')
    for name in benchmark_names:
        LOGGER.info('{}'.format(str(name)))

    # Ok, run all requested benchmarks
    for name in benchmark_names:
        benchmark = BENCHMARK_DICT[name]
        LOGGER.info('Running %s benchmark', benchmark.name)

        benchmark_data_dir = data_dir / benchmark.name.replace('-', '_')

        if not benchmark_data_dir.exists():
            LOGGER.error('Data directory {} does not exist! Is the data for benchmark {} downloaded?'.format(str(benchmark_data_dir), name))

            if skip:
                LOGGER.error('Skipping benchmark {}'.format(name))
                continue
            else:
                click.abort()

        try:
            ctx.invoke(benchmark, data_dir=benchmark_data_dir, model_dir=model_dir)
        except Exception as e:
            LOGGER.error('Failed to run benchmark {} due to unhandled exception.\n{}'.format(name, e))

            if skip:
                LOGGER.error('Skipping benchmark {}'.format(name))
                continue
            else:
                click.abort()

@cli.command(help='Display system information')
def sysinfo():
    from sciml_bench.core.system import HostSpec, DeviceSpecs, bytesto

    host_spec = HostSpec()

    click.echo('------------------')
    click.echo('Host')
    click.echo('------------------')
    click.echo('Node name: {}'.format(host_spec.node_name))
    click.echo('IP name: {}'.format(host_spec.ip_address))
    click.echo('System name: {}'.format(host_spec.system))
    click.echo('System release: {}'.format(host_spec.release))
    click.echo('No. of cores: {}'.format(host_spec.num_cores))
    click.echo('Total Memory: {:.2f}GB'.format(bytesto(host_spec.total_memory, 'g')))
    for name, value in host_spec.cpu_info.items():
        name = name.replace('_', ' ')
        name = string.capwords(name)
        name = name.replace('Cpu', 'CPU')
        click.echo('{}: {}'.format(name, value))

    device_spec = DeviceSpecs()
    click.echo('------------------')
    click.echo('Devices')
    click.echo('------------------')
    click.echo('Found {} device{}'.format(device_spec.device_count, 's' if device_spec.device_count > 1 else ''))

    for number in range(device_spec.device_count):
        name = 'gpu_{}'.format(number)
        click.echo('------------------')
        click.echo('Device {}'.format(number))
        click.echo('------------------')
        click.echo('Device name: {}'.format(device_spec.names[name + '_name']))
        memory = device_spec.memory[name + '_memory_total']
        click.echo('Total Memory: {:.2f}GB'.format(bytesto(memory, 'g')))



@cli.command(help='Download benchmark datasets from remote store')
@click.argument('name', type=click.Choice(['all', 'em_denoise', 'dms_classifier', 'slstr_cloud']))
@click.argument('destination')
@click.option('--user', default=None, help='Username to use to login to remote data store')
def download(*args, **kwargs):
    download_datasets(*args, **kwargs)


def _run_benchmark(module, ctx, **kwargs):
    benchmark_name = ctx.command.name.replace('-', '_')

    now = datetime.now()
    folder = now.strftime("%Y-%m-%d-%H%M")

    kwargs.update(ctx.obj)
    kwargs['data_dir'] = str(Path(kwargs['data_dir']) / benchmark_name)
    kwargs['model_dir'] = str(Path(kwargs['model_dir']).joinpath(benchmark_name).joinpath(folder))
    kwargs['metrics'] = list(kwargs['metrics'])
    module.main(**kwargs)

@cli.command(help='Generate report from benchmark runs')
@click.pass_context
def report(ctx, *args, **kwargs):
    kwargs.update(ctx.obj)
    model_dir = kwargs.get('model_dir')

    for benchmark in BENCHMARKS:
        benchmark_folder = Path(model_dir) / benchmark.name.replace('-', '_')
        if not benchmark_folder.exists():
            continue

        folders = [x for x in benchmark_folder.iterdir() if x.is_dir()]

        for folder in folders:
            create_report(folder)


if __name__ == "__main__":
    cli(auto_envvar_prefix='SCIML_BENCH')
