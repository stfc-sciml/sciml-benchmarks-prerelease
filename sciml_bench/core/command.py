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
    for node_name, local_size in data:
        LOGGER.info('%s has %s processes', node_name, local_size)

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
@click.option('--data-dir', default='data', help='Data directory location', envvar='SCIML_BENCH_DATA_DIR')
@click.option('--model-dir', default='sciml-bench-out', type=str, help='Output directory for model results', envvar='SCIML_BENCH_MODEL_DIR')
@click.option('--lr-warmup', default=3, type=int, help='Number of epochs over which to scale the learning rate.')
@click.option('--cpu-only', default=False, is_flag=True, help='Disable GPU execution')
@click.option('--use-amp', default=False, is_flag=True, help='Enable Automatic Mixed Precision')
@click.option('--exec-mode', default='train_and_predict', type=click.Choice(['train', 'train_and_predict', 'predict']), help='Set the execution mode')
@click.option('--log-batch', default=False, is_flag=True, help='Whether to log metrics by batch or by epoch')
@click.option('--log-interval', default=0.5, help='Logging interval for system metrics')
@click.option('--seed', default=42, type=int, help='Random seed to use for initialization of random state')
@click.option('--log-level', default=logging.INFO, type=int, help='Log level to use for printing to stdout')
def cli(ctx, tracking_uri=None, **kwargs):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    with warnings.catch_warnings():
        hvd.init()
        ctx.ensure_object(dict)
        ctx.obj.update(kwargs)

        print_header()
        set_environment_variables(**kwargs)

@cli.command('list', help='List benchmarks')
@click.pass_context
def cmd_list(ctx, **kwargs):
    click.echo('Available Benchmarks\n')

    for benchmark in BENCHMARKS:
        path = Path(ctx.obj['data_dir']).joinpath(benchmark.name.replace('-', '_'))
        downloaded = path.exists()
        click.echo('{}\t\tDownloaded: {}'.format(benchmark.name, downloaded))

@cli.command(help='Run all benchmarks with default settings')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
def all(ctx, data_dir, model_dir, **params):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)

    if not data_dir.exists():
        click.echo("Data directory {} does not exist!".format(data_dir))
        sys.exit()

    for benchmark in BENCHMARKS:
        LOGGER.info('Running %s benchmark', benchmark.name)
        ctx.invoke(benchmark, data_dir=data_dir / benchmark.name, model_dir=model_dir)

@cli.command(help='Run the DMS Classifier Benchmark')
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

@cli.command(help='Run the Electron Microscopy Denoise Benchmark')
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

@cli.command(help='Run the SLSTR Cloud Segmentation Benchmark')
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

BENCHMARKS = [
    dms_classifier,
    em_denoise,
    slstr_cloud
]

if __name__ == "__main__":
    cli(auto_envvar_prefix='SCIML_BENCH')
