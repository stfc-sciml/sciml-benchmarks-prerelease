import yaml
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from pathlib import Path

from sciml_bench.core.dllogger import tags, LOGGER
from sciml_bench.core.utils.hooks.mlflow import MLFlowCallback
from sciml_bench.core.utils.hooks.profiling_hook import ProfilingHook

class Benchmark:

    def __init__(self, model_fn, dataset):
        self._model = None
        self._model_fn = model_fn
        self._dataset = dataset
        self._results = {}

    def build(self, **params):
        self._model = self._model_fn(self._dataset.dimensions, **params)

    def train(self, epochs=1, **params):
        if self._model is None:
            raise RuntimeError("Model has not been built!\n \
                    Please call benchmark.build() first to compile the model!")

        spe = int(np.ceil(self._dataset.train_size / params['global_batch_size']))

        hooks = []

        # Add hook for capturing Img/s and duration
        train_profiler_hook = ProfilingHook(params['batch_size'],
                                            spe, num_replicas=params['num_replicas'])
        hooks.append(train_profiler_hook)

        mlf_callback = MLFlowCallback()
        hooks.append(mlf_callback)

        # Add hook for capturing metrics vs. epoch
        log_file = Path(params['model_dir']).joinpath('training.log')
        csv_logger = tf.keras.callbacks.CSVLogger(log_file)
        hooks.append(csv_logger)

        LOGGER.log('Begin Training...')
        LOGGER.log('Training for {} epochs'.format(epochs))
        LOGGER.log('Epoch contains {} steps'.format(spe))


        dataset = self._dataset.train_fn(params['global_batch_size'])

        LOGGER.log(tags.RUN_START)

        self._model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=spe,
            callbacks=hooks)

        LOGGER.log(tags.RUN_STOP)

        self._results['train'] = train_profiler_hook.get_results()

    def predict(self, **params):
        if self._model is None:
            raise RuntimeError("Model has not been built!\n \
                    Please call benchmark.build() first to compile the model!")

        test_profiler_hook = ProfilingHook(params['batch_size'],
                                           warmup_steps=5, num_replicas=params['num_replicas'])

        hooks = [test_profiler_hook]

        mlf_callback = MLFlowCallback()
        hooks.append(mlf_callback)

        predict_steps = int(np.ceil(self._dataset.test_size / params['global_batch_size']))

        LOGGER.log('Begin Predict...')
        LOGGER.log('Predicting for {} steps'.format(predict_steps))
        LOGGER.log(tags.RUN_START)

        dataset = self._dataset.test_fn(params['global_batch_size'])
        metrics = self._model.evaluate(dataset, steps=predict_steps, callbacks=hooks)
        # Handle special case: only metric is the loss
        metrics = metrics if isinstance(metrics, list) else [metrics]
        metrics = {name: float(value) for name, value in zip(self._model.metrics_names, metrics)}

        LOGGER.log(tags.RUN_STOP)
        LOGGER.log("Predict finished")

        self._results['test'] = test_profiler_hook.get_results()
        self._results['test'].update(metrics)

    def save_results(self, **params):
        model_dir = params['model_dir']
        results_file = Path(model_dir).joinpath('results.yml')

        with results_file.open('w') as handle:
            yaml.dump(self._results, handle)

        params_file = Path(model_dir).joinpath('params.yml')
        with params_file.open('w') as handle:
            yaml.dump(params, handle)

        weights_file = str(Path(model_dir).joinpath('final_weights.h5'))
        self._model.save_weights(weights_file)



