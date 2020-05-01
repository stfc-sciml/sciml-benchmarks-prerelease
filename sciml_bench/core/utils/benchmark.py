import numpy as np
import tensorflow as tf
import mlflow
from pathlib import Path
import horovod.tensorflow.keras as hvd
from sciml_bench.core.dllogger import tags, LOGGER
from sciml_bench.core.utils.hooks.mlflow import MLFlowCallback, MLFlowTimingCallback


class MultiNodeBenchmark:

    def __init__(self, model_fn, dataset):
        self._model = None
        self._model_fn = model_fn
        self._dataset = dataset
        self._results = {}

    def build(self, log_batch=False, loss=tf.losses.BinaryCrossentropy(), learning_rate=0.001, metrics=['accuracy'], **params):
        self._log_batch = log_batch

        # Horovod: adjust learning rate based on number of GPUs.

        self._model = self._model_fn(self._dataset.dimensions, **params)

        opt = tf.optimizers.Adam(learning_rate * hvd.size())
        opt = hvd.DistributedOptimizer(opt)

        self._model.compile(loss=loss,
                    optimizer=opt,
                    metrics=metrics,
                    experimental_run_tf_function=False)

    def train(self, epochs=1, lr_warmup=3, **params):

        verbose = 1 if hvd.rank() == 0 else 0

        if self._model is None:
            raise RuntimeError("Model has not been built!\n \
                    Please call benchmark.build() first to compile the model!")

        spe = int(np.ceil(self._dataset.train_size / params['global_batch_size']))

        # Add hooks for Horovod
        hooks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(steps_per_epoch=spe, warmup_epochs=lr_warmup, verbose=1),
        ]

        if hvd.rank() == 0:
            # These hooks only need to be called by one instance.
            # Therefore we need to only add them on rank == 0
            mlf_callback = MLFlowCallback(self._log_batch)
            hooks.append(mlf_callback)

            # Add hook for capturing Img/s and duration
            profiler_hook = MLFlowTimingCallback(params['global_batch_size'])
            hooks.append(profiler_hook)

        # Add hook for capturing metrics vs. epoch
        log_file = Path(params['model_dir']).joinpath('training.log')
        csv_logger = tf.keras.callbacks.CSVLogger(log_file)
        hooks.append(csv_logger)

        LOGGER.log('Begin Training...')
        LOGGER.log('Training for {} epochs'.format(epochs))
        LOGGER.log('Epoch contains {} steps'.format(spe))

        dataset = self._dataset.train_fn(params['batch_size'])

        LOGGER.log(tags.RUN_START)

        self._model.fit(dataset,
                epochs=epochs,
                steps_per_epoch=spe,
                callbacks=hooks,
                verbose=verbose)

        LOGGER.log(tags.RUN_STOP)

        if hvd.rank() == 0:
            model_dir = Path(params['model_dir'])
            model_dir.mkdir(parents=True, exist_ok=True)
            weights_file = str(model_dir / 'final_weights.h5')
            self._model.save_weights(weights_file)
            mlflow.log_artifact(weights_file)

    def predict(self, lr_warmup=3, **params):
        if self._model is None:
            raise RuntimeError("Model has not been built!\n \
                    Please call benchmark.build() first to compile the model!")

        predict_steps = int(np.ceil(self._dataset.test_size / params['global_batch_size']))

        # Add hooks for Horovod
        hooks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(steps_per_epoch=predict_steps, warmup_epochs=lr_warmup, verbose=1),
        ]

        if hvd.rank() == 0:
            # These hooks only need to be called by one instance.
            # Therefore we need to only add them on rank == 0
            mlf_callback = MLFlowCallback(self._log_batch)
            hooks.append(mlf_callback)

            profiler_hook = MLFlowTimingCallback(params['global_batch_size'])
            hooks.append(profiler_hook)

        LOGGER.log('Begin Predict...')
        LOGGER.log('Predicting for {} steps'.format(predict_steps))
        LOGGER.log(tags.RUN_START)

        dataset = self._dataset.test_fn(params['batch_size'])
        verbose = 1 if hvd.rank() == 0 else 0
        metrics = self._model.evaluate(dataset, steps=predict_steps, callbacks=hooks, verbose=verbose)

        # Handle special case: only metric is the loss
        metrics = metrics if isinstance(metrics, list) else [metrics]
        metrics = {name: float(value) for name, value in zip(self._model.metrics_names, metrics)}

        LOGGER.log(tags.RUN_STOP)
        LOGGER.log("Predict finished")
