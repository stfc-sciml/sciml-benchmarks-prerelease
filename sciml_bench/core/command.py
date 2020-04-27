import os

# Optimization flags
os.environ['CUDA_CACHE_DISABLE'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'

os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

import sys
import yaml
import click
import click_config_file
import mlflow
from pathlib import Path
import horovod.tensorflow as hvd

from sciml_bench.core.utils.hooks.mlflow import DistributedMLFlowRun
from sciml_bench.core.dllogger.logger import LOGGER
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

def set_environment_variables(cpu_only=False, use_amp=False, **kwargs):
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


@click.group()
@click.pass_context
@click.option('--tracking-uri', default=None, type=str, help='Tracking URI for MLFlow', envvar='SCIML_BENCH_TRACKING_URI')
@click.option('--lr-warmup', default=3, type=int, help='Number of epochs over which to scale the learning rate.')
@click.option('--cpu-only', default=False, is_flag=True, help='Disable GPU execution')
@click.option('--use-amp', default=False, is_flag=True, help='Enable Automatic Mixed Precision')
@click.option('--exec-mode', default='train_and_predict', type=click.Choice(['train', 'train_and_predict', 'predict']), help='Set the execution mode')
@click.option('--log-batch', default=False, is_flag=True, help='Whether to log metrics by batch or by epoch')
@click.option('--log-interval', default=0.5, help='Logging interval for system metrics')
@click.option('--seed', default=42, type=int, help='Random seed to use for initialization of random state')
def cli(ctx, tracking_uri=None, **kwargs):
    hvd.init()
    ctx.ensure_object(dict)
    ctx.obj.update(kwargs)

    mlflow.set_tracking_uri(tracking_uri)
    print_header()
    set_environment_variables(**kwargs)


@cli.command(help='Run all benchmarks with default settings')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.option('--data-dir', default='data', help='Data directory location', envvar='SCIML_BENCH_DATA_DIR')
@click.option('--model-dir', default='sciml-bench-out', type=str, help='Output directory for model results', envvar='SCIML_BENCH_MODEL_DIR')
def all(ctx, data_dir, model_dir, **params):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)

    if not data_dir.exists():
        click.echo("Data directory {} does not exist!".format(data_dir))
        sys.exit()

    LOGGER.info("Running DMS Classifier Benchmark")
    ctx.invoke(dms_classifier, data_dir=data_dir / 'dms_classifier', model_dir=model_dir)

    LOGGER.info("Running EM Denoise Benchmark")
    ctx.invoke(em_denoise, data_dir=data_dir / 'em_denoise', model_dir=model_dir)

    LOGGER.info("Running SLSTR Cloud Segmentation Benchmark")
    ctx.invoke(slstr_cloud, data_dir=data_dir / 'slstr_cloud', model_dir=model_dir)

@cli.command(help='Run the DMS Classifier Benchmark')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.option('--data-dir', default='data/dms_classifier', envvar='SCIML_BENCH_DATA_DIR')
@click.option('--model-dir', default='sciml-bench-out', envvar='SCIML_BENCH_MODEL_DIR')
@click.option('--epochs', default=10, help='Set number of epochs')
@click.option('--loss', default='binary_crossentropy', help='Set loss function to use')
@click.option('--batch-size', default=256, help='Set the batch size for training & test')
@click.option('--learning-rate', default=1e-4, help='Set the learning rate')
@click.option('--metrics', '-m', default=['accuracy'], multiple=True, help='Set the metrics to output')
def dms_classifier(ctx, **kwargs):
    import sciml_bench.dms_classifier.main as dms_classifier_mod
    mlflow.set_experiment('dms_classifier')

    with DistributedMLFlowRun():
        kwargs.update(ctx.obj)
        kwargs['model_dir'] = str(Path(kwargs['model_dir']) / 'dms_classifier')
        kwargs['metrics'] = list(kwargs['metrics'])
        dms_classifier_mod.main(**kwargs)

@cli.command(help='Run the Electron Microscopy Denoise Benchmark')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.option('--data-dir', default='data/em_denoise', envvar='SCIML_BENCH_DATA_DIR')
@click.option('--model-dir', default='sciml-bench-out', envvar='SCIML_BENCH_MODEL_DIR')
@click.option('--epochs', default=10, help='Set number of epochs')
@click.option('--loss', default='mse', help='Set loss function to use')
@click.option('--batch-size', default=256, help='Set the batch size for training & test')
@click.option('--learning-rate', default=0.01, help='Set the learning rate')
@click.option('--metrics', '-m', default=[], multiple=True, help='Set the metrics to output')
def em_denoise(ctx, **kwargs):
    import sciml_bench.em_denoise.main as em_denoise_mod

    mlflow.set_experiment('em_denoise')

    with DistributedMLFlowRun():
        kwargs.update(ctx.obj)
        kwargs['model_dir'] = str(Path(kwargs['model_dir']) / 'em_denoise')
        kwargs['metrics'] = list(kwargs['metrics'])
        em_denoise_mod.main(**kwargs)


@cli.command(help='Run the SLSTR Cloud Segmentation Benchmark')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.option('--data-dir', default='data/slstr_cloud', envvar='SCIML_BENCH_DATA_DIR')
@click.option('--model-dir', default='sciml-bench-out', envvar='SCIML_BENCH_MODEL_DIR')
@click.option('--epochs', default=30, help='Set number of epochs')
@click.option('--loss', default='binary_crossentropy', help='Set loss function to use')
@click.option('--batch-size', default=6, help='Set the batch size for training & test')
@click.option('--learning-rate', default=0.001, help='Set the learning rate')
@click.option('--metrics', '-m', default=['accuracy'], multiple=True, help='Set the metrics to output')
def slstr_cloud(ctx, **kwargs):
    import sciml_bench.slstr_cloud.main as slstr_cloud_mod
    mlflow.set_experiment('slstr_cloud')

    with DistributedMLFlowRun():
        kwargs.update(ctx.obj)
        kwargs['model_dir'] = str(Path(kwargs['model_dir']) / 'slstr_cloud')
        kwargs['metrics'] = list(kwargs['metrics'])
        slstr_cloud_mod.main(**kwargs)

@cli.command(help='Download benchmark datasets from remote store')
@click.argument('name', type=click.Choice(['all', 'em_denoise', 'dms_classifier', 'slstr_cloud']))
@click.argument('destination')
@click.option('--user', default=None, help='Username to use to login to remote data store')
def download(*args, **kwargs):
    download_datasets(*args, **kwargs)


if __name__ == "__main__":
    cli(auto_envvar_prefix='SCIML_BENCH')
