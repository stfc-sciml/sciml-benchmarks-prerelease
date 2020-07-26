import os
import traceback
import string
import logging
import sys
import yaml
import click
from pathlib import Path

import sciml_bench
from sciml_bench.core.logging import LOGGER
from sciml_bench.benchmarks import register_all_objects
from sciml_bench.core.report import create_report
from sciml_bench.core.download import download_datasets
from sciml_bench.core.runner import run_benchmark
from sciml_bench.benchmarks import BENCHMARKS


def register():
    config = load_yaml('config.yml')
    # Find all & import all modules in search path to register models with sciml_bench
    register_all_objects()
    for path in config.get('search_path', []):
        register_all_objects(path)


def load_yaml(file_path):
    if not Path(file_path).exists():
        return {}

    with open(file_path) as config_data:
        cfg = yaml.load(config_data, yaml.SafeLoader)
        cfg = {} if cfg is None else cfg
        return cfg


def print_header():
    import horovod.tensorflow as hvd
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
    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

    os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    if kwargs['verbosity'] >= 3 and kwargs['log_level'] == 'debug':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'

    # Try and import tensorflow to check for any issues
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices('GPU')
        if len(devices) == 0 and not cpu_only:
            LOGGER.warning('No available GPUs could be detected. This could be because no GPU exists or could be due to a mismatch between CUDA runtime version and the compute capability of the system hardware. Check that the CUDA drivers are correctly installted on your system. Set verbosity = 3 and check the output of the Tensorflow logs.')
        #     sys.exit(1)
    except Exception as e:
        LOGGER.debug(traceback.format_exc())
        LOGGER.critical('Fatal issue importing Tensorflow: %s', e)
        sys.exit(1)


@click.group()
def cli(tracking_uri=None, **kwargs):
    pass


@cli.command('list', help='List benchmarks')
@click.argument('name', default='all', type=click.Choice(['all', 'benchmarks', 'datasets']))
@click.option('--data-dir', default='data', help='Data directory location', envvar='SCIML_BENCH_DATA_DIR')
@click.pass_context
def cmd_list(ctx, name, data_dir):
    register()

    if name == 'benchmarks' or name == 'all':
        click.echo('Benchmarks\n')

        for benchmark in BENCHMARKS:
            click.echo(benchmark)

    if name == 'datasets' or name == 'all':
        click.echo('')
        click.echo('Datasets\n')

        for benchmark in BENCHMARKS:
            path = Path(data_dir).joinpath(benchmark)
            downloaded = path.exists()
            click.echo('{}\t\tDownloaded: {}'.format(benchmark, downloaded))


@cli.command(help='Run SciML benchmarks')
@click.argument('benchmark_names', nargs=-1)
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
@click.option('--log-level', default='info', type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']), help='Log level to use for printing to stdout')
@click.option('--skip/--no-skip', default=True, help='Whether to skip or exit on encountering an exception')
def run(benchmark_names, skip, **params):
    # Load configuration for benchmarks
    config = load_yaml('config.yml')

    LOGGER.setLevel(params.get('log_level').upper())
    if params.get('verbosity') < 2:
        LOGGER.setLevel(logging.WARNING)
    if params.get('verbosity') == 0:
        LOGGER.setLevel(logging.CRITICAL)

    set_environment_variables(**params)

    if params.get('verbosity') >= 2:
        print_header()

    register()

    for name in benchmark_names:
        if name not in BENCHMARKS:
            LOGGER.error('No benchmark with name {}'.format(name))
            sys.exit(1)

    model_dir = params['model_dir']
    data_dir = params['data_dir']

    data_dir = Path(data_dir)

    if not data_dir.exists():
        LOGGER.error("Data directory {} does not exist!".format(data_dir))
        sys.exit(1)

    LOGGER.info('Model directory is: %s', str(model_dir))
    LOGGER.info('Data directory is: %s', str(data_dir))

    # If no benchmarks specified or all then run everything
    if len(benchmark_names) == 0 or 'all' in benchmark_names:
        benchmark_names = BENCHMARKS.keys()

    # Sanity check: do all the benchmarks exist?
    for name in benchmark_names:
        if name not in BENCHMARKS:
            LOGGER.error('Benchmark {} does not exist!'.format(name))
            click.Abort()

    # Log which benchmarks we will run
    LOGGER.info('Selected the following benchmarks:')
    for name in benchmark_names:
        LOGGER.info('{}'.format(str(name)))

    # Ok, run all requested benchmarks
    for name in benchmark_names:

        LOGGER.info('Running %s benchmark', name)

        benchmark_data_dir = data_dir / name

        if not benchmark_data_dir.exists():
            LOGGER.error('Data directory {} does not exist! Is the data for benchmark {} downloaded?'.format(str(benchmark_data_dir), name))

            if skip:
                LOGGER.error('Skipping benchmark {}'.format(name))
                continue
            else:
                sys.exit(1)

        cfg = dict(config[name]) if name in config else {}
        cfg.update(params)
        cfg['data_dir'] = benchmark_data_dir

        benchmark = BENCHMARKS[name](**cfg)

        try:
            run_benchmark(benchmark, **cfg)
        except Exception as e:
            LOGGER.debug(traceback.format_exc())
            LOGGER.error('Failed to run benchmark {} due to unhandled exception.\n{}'.format(name, e))

            if skip:
                LOGGER.info('Skipping benchmark {}'.format(name))
                continue
            else:
                sys.exit(1)


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


@cli.command(help='Generate report from benchmark runs')
@click.option('--model-dir', default='sciml-bench-out', type=str, help='Output directory for model results', envvar='SCIML_BENCH_MODEL_DIR')
@click.pass_context
def report(ctx, model_dir, **kwargs):
    register()
    for benchmark in BENCHMARKS:
        benchmark_folder = Path(model_dir) / benchmark
        if not benchmark_folder.exists():
            continue

        folders = [x for x in benchmark_folder.iterdir() if x.is_dir()]

        for folder in folders:
            create_report(folder)


if __name__ == "__main__":
    cli(auto_envvar_prefix='SCIML_BENCH')
