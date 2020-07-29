import tensorflow as tf
from pathlib import Path
from datetime import datetime
import horovod.tensorflow.keras as hvd

from sciml_bench.core.bench_logger import LOGGER
from sciml_bench.core.tracking import TrackingClient
from sciml_bench.core.system import HostSpec, DeviceSpecs
from sciml_bench.core.callbacks import NodeLogger
from sciml_bench.core.callbacks import TrackingCallback
from sciml_bench.core.benchmark import TensorflowKerasMixin


class BenchmarkRunner:

    def __init__(self, benchmark, output_dir):
        self._benchmark = benchmark
        self._output_dir = output_dir

        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        host_spec = HostSpec()
        self._node_name = host_spec.node_name

        # Log system information if on local rank 0
        if hvd.local_rank() == 0:

            # Log host information
            file_name = '{}_host.json'.format(self._node_name)
            db = TrackingClient(Path(self._output_dir) / file_name)

            host_info = {
                'name': host_spec.name,
                'node_name': host_spec.node_name,
                'ip': host_spec.node_name,
                'num_cores': host_spec.num_cores,
                'release': host_spec.release,
                'system': host_spec.system,
                'cpu_info': host_spec.cpu_info,
            }

            db.log_tag('host_info', host_info)

            # Log device information
            device_specs = DeviceSpecs()

            file_name = '{}_devices.json'.format(self._node_name)
            db = TrackingClient(Path(self._output_dir) / file_name)

            device_info = {}
            device_info['gpu_count'] = device_specs.device_count
            device_info.update({'gpu_{}'.format(i): device_specs.get_device_info(i) for i in range(device_specs.device_count)})

            db.log_tag('device_info', device_info)

    @property
    def benchmark(self):
        return self._benchmark

    def setup(self, **params):
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                LOGGER.warning('Could not set GPU memory growth == True')

        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        params['num_replicas'] = hvd.size()
        num_replicas = params['num_replicas']
        params['global_batch_size'] = params['batch_size'] * num_replicas

        return params

    def run(self, log_interval=0.5, **params):

        params = self.setup(**params)
        self.build(**params)

        if hvd.rank() == 0:
            db = TrackingClient(Path(self._output_dir) / 'logs.json')
            db.log_param('params', params)

        LOGGER.info('Number of Replicas: {}'.format(params['num_replicas']))
        LOGGER.info('Global Batch Size: {}'.format(params['global_batch_size']))
        LOGGER.info('Replica Batch Size: {}'.format(params['batch_size']))

        if 'train' in params['exec_mode']:
            with NodeLogger(self._output_dir, name=self._node_name, prefix='train', interval=log_interval):
                self.train(**params)

        if 'predict' in params['exec_mode']:
            with NodeLogger(self._output_dir, name=self._node_name, prefix='predict', interval=log_interval):
                self.predict(**params)


class TensorflowKerasBenchmarkRunner(BenchmarkRunner):

    def __init__(self, benchmark, output_dir):
        super().__init__(benchmark, output_dir=output_dir)

    def build(self, log_batch=False, **params):
        self._log_batch = log_batch

        self._model = self.benchmark.model(input_shape=self.benchmark.data_loader_.input_shape, **params)

        opt = self.benchmark.optimizer_
        opt_cfg = opt.get_config()
        opt_cfg['learning_rate'] *= hvd.size()
        opt = opt.from_config(opt_cfg)
        opt = hvd.DistributedOptimizer(opt)

        loss = self.benchmark.loss_
        LOGGER.debug(loss.__name__)
        metrics = self.benchmark.metrics

        self._model.compile(loss=loss,
                    optimizer=opt,
                    metrics=metrics,
                    experimental_run_tf_function=False)

    def train(self, **params):
        verbose = 1 if params.get('verbosity', 0) > 1 and hvd.rank() == 0 else 0

        if self._model is None:
            raise RuntimeError("Model has not been built!\n \
                    Please call benchmark.build() first to compile the model!")

        # Add hooks for Horovod
        hooks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ]

        if hvd.rank() == 0:
            # These hooks only need to be called by one instance.
            # Therefore we need to only add them on rank == 0
            tracker_hook = TrackingCallback(params['model_dir'], params['global_batch_size'], self._log_batch)
            hooks.append(tracker_hook)

        # Add hook for capturing metrics vs. epoch
        log_file = Path(params['model_dir']).joinpath('training.log')
        csv_logger = tf.keras.callbacks.CSVLogger(log_file)
        hooks.append(csv_logger)

        LOGGER.info('Begin Training...')
        LOGGER.info('Training for {} epochs'.format(self.benchmark.epochs))

        dataset = self.benchmark.data_loader_.to_dataset()

        LOGGER.debug('Fitting Start')

        self._model.fit(dataset,
                epochs=self.benchmark.epochs,
                callbacks=hooks,
                verbose=verbose, **self.benchmark.fit_params)

        LOGGER.debug('Fitting End')

        if hvd.rank() == 0:
            model_dir = Path(params['model_dir'])
            model_dir.mkdir(parents=True, exist_ok=True)
            weights_file = str(model_dir / 'final_weights.h5')
            self._model.save_weights(weights_file)

    def predict(self, lr_warmup=3, **params):
        if self._model is None:
            raise RuntimeError("Model has not been built!\n \
                    Please call benchmark.build() first to compile the model!")

        # Add hooks for Horovod
        hooks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ]

        if hvd.rank() == 0:
            # These hooks only need to be called by one instance.
            # Therefore we need to only add them on rank == 0
            tracker_hook = TrackingCallback(params['model_dir'], params['global_batch_size'], self._log_batch)
            hooks.append(tracker_hook)

        LOGGER.info('Begin Predict...')

        dataset = self.benchmark.validation_data_loader_.to_dataset()
        verbose = 1 if params.get('verbosity', 0) > 1 and hvd.rank() == 0 else 0

        LOGGER.debug('Evaluate Start')
        self._model.evaluate(dataset, callbacks=hooks, verbose=verbose)
        LOGGER.debug('Evaluate End')


def run_benchmark(benchmark, **params):
    benchmark_name = benchmark.name

    now = datetime.now()
    folder = now.strftime("%Y-%m-%d-%H%M")

    params['data_dir'] = Path(params['data_dir']) / benchmark_name
    params['model_dir'] = str(Path(params['model_dir']).joinpath(benchmark_name).joinpath(folder))
    params['metrics'] = list(benchmark.metrics)
    params['batch_size'] = benchmark.batch_size

    if not isinstance(benchmark, TensorflowKerasMixin):
        raise RuntimeError("Expected benchmark to be a tensorflow model but it was not!")

    LOGGER.debug('Benchmark %s', benchmark.name)
    LOGGER.debug('Loss %s', benchmark.loss)
    LOGGER.debug('Batch size %s', benchmark.batch_size)
    LOGGER.debug('Optimizer %s', benchmark.optimizer)
    LOGGER.debug('Epochs %s', benchmark.epochs)

    runner = TensorflowKerasBenchmarkRunner(benchmark, output_dir=params['model_dir'])
    runner.run(**params)
