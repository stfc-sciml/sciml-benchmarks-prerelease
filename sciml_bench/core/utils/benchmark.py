import json
import tensorflow as tf
from pathlib import Path

from sciml_bench.core.dllogger import tags, LOGGER
from sciml_bench.core.utils.hooks.profiling_hook import ProfilingHook

class Benchmark:

    def __init__(self, model, dataset):
        self._model = model
        self._dataset = dataset
        self._results = {}

    def train(self, params):
        spe = self._dataset.train_size / params['global_batch_size']

        hooks = []

        # Add hook for capturing Img/s and duration
        train_profiler_hook = ProfilingHook(params['batch_size'],
                                            spe, num_replicas=params['num_replicas'])
        hooks.append(train_profiler_hook)

        # Add hook for capturing metrics vs. epoch
        log_file = Path(params['model_dir']).joinpath('training.log')
        csv_logger = tf.keras.callbacks.CSVLogger(log_file)
        hooks.append(csv_logger)


        LOGGER.log('Begin Training...')
        LOGGER.log('Training for {} epochs'.format(params['epochs']))
        LOGGER.log('Epoch contains {} steps'.format(spe))

        dataset = self._dataset.train_fn()

        LOGGER.log(tags.RUN_START)
        self._model.fit(
            dataset,
            epochs=params['epochs'],
            steps_per_epoch=spe,
            callbacks=hooks)
        LOGGER.log(tags.RUN_STOP)

        self._results['train'] = train_profiler_hook.get_results()

    def predict(self, params):
        test_profiler_hook = ProfilingHook(params['batch_size'],
                                           warmup_steps=2, num_replicas=params['num_replicas'])
        hooks = [test_profiler_hook]

        predict_steps = self._dataset.test_size / params['global_batch_size']

        LOGGER.log('Begin Predict...')
        LOGGER.log('Predicting for {} steps'.format(predict_steps))
        LOGGER.log(tags.RUN_START)

        dataset = self._dataset.test_fn()
        metrics = self._model.evaluate(dataset, steps=predict_steps, callbacks=hooks)
        metrics = {name: float(value) for name, value in zip(self._model.metrics_names, metrics)}

        LOGGER.log(tags.RUN_STOP)
        LOGGER.log("Predict finished")

        self._results['test'] = test_profiler_hook.get_results()
        self._results['test'].update(metrics)

    def save_results(self, model_dir):
        results_file = Path(model_dir).joinpath('results.json')
        with results_file.open('w') as handle:
            json.dump(self._results, handle)

        weights_file = str(Path(model_dir).joinpath('final_weights'))
        self._model.save_weights(weights_file)

