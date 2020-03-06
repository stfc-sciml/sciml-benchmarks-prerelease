import os
import yaml
import click
import click_config_file

import sciml_bench.dms_classifier.main as dms_classifier_mod
import sciml_bench.em_denoise.main as em_denoise_mod
import sciml_bench.slstr_cloud.main as slstr_cloud_mod

def yaml_provider(file_path, cmd_name):
    with open(file_path) as config_data:
        cfg = yaml.load(config_data, yaml.SafeLoader)
        return cfg

def set_environment_variables(use_amp=False, **kwargs):
    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

    os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

    if use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


@click.group()
@click.pass_context
@click.option('--lr-scaling', default='none', type=click.Choice(['linear', 'none']), help='How to scale the learning rate at larger batch sizes')
@click.option('--use-amp', default=False, is_flag=True, help='Enable Automatic Mixed Precision')
@click.option('--exec-mode', default='train_and_predict', type=click.Choice(['train', 'train_and_predict', 'predict']), help='Set the execution mode')
@click.option('--seed', default=42, type=int, help='Random seed to use for initialization of random state')
def cli(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj.update(kwargs)
    set_environment_variables(**kwargs)

@cli.command(help='Run the DMS Classifier Benchmark')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.argument('data_dir')
@click.argument('model_dir')
@click.option('--epochs', default=50, help='Set number of epochs')
@click.option('--batch_size', default=32, help='Set the batch size for training & test')
@click.option('--learning-rate', default=1e-4, help='Set the learning rate')
def dms_classifier(ctx, **kwargs):
    kwargs.update(ctx.obj)
    dms_classifier_mod.main(**kwargs)

@cli.command(help='Run the Electron Microscopy Denoise Benchmark')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.argument('data_dir')
@click.argument('model_dir')
@click.option('--epochs', default=10, help='Set number of epochs')
@click.option('--batch_size', default=10, help='Set the batch size for training & test')
@click.option('--learning-rate', default=0.01, help='Set the learning rate')
def em_denoise(ctx, **kwargs):
    kwargs.update(ctx.obj)
    em_denoise_mod.main(**kwargs)

@cli.command(help='Run the SLSTR Cloud Segmentation Benchmark')
@click_config_file.configuration_option(provider=yaml_provider, implicit=False)
@click.pass_context
@click.argument('data_dir')
@click.argument('model_dir')
@click.option('--epochs', default=30, help='Set number of epochs')
@click.option('--batch_size', default=8, help='Set the batch size for training & test')
@click.option('--learning-rate', default=0.001, help='Set the learning rate')
def slstr_cloud(ctx, **kwargs):
    kwargs.update(ctx.obj)
    slstr_cloud_mod.main(**kwargs)


if __name__ == "__main__":
    cli(params={})
