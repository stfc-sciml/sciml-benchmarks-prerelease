import time
from itertools import chain
from sciml_bench.core.test.helpers import fake_model_fn, FakeDataLoader
from sciml_bench.core.utils.hooks.mlflow import MLFlowDeviceLogger, MLFlowHostLogger, MLFlowCallback, log_device_stats, log_host_stats, MLFlowTimingCallback

def test_MLFlowDeviceLogger(mocker):

    logger = MLFlowDeviceLogger(interval=0.1)

    run_stub = mocker.spy(logger, 'run')

    logger.start()
    time.sleep(1)
    logger.stop()

    run_stub.assert_called()

def test_MLFlowHostLogger(mocker):

    logger = MLFlowHostLogger(interval=0.1)

    run_stub = mocker.spy(logger, 'run')

    logger.start()
    time.sleep(1)
    logger.stop()

    run_stub.assert_called()

def test_log_device_stats(mocker):

    @log_device_stats(name='my_func', interval=0.1)
    def my_func():
        time.sleep(1)

    spy = mocker.spy(MLFlowDeviceLogger, 'run')
    my_func()
    spy.assert_called()

def test_log_host_stats(mocker):

    @log_host_stats(name='my_func', interval=0.1)
    def my_func():
        time.sleep(1)

    spy = mocker.spy(MLFlowHostLogger, 'run')
    my_func()
    spy.assert_called()

def test_MLFlowCallback(mocker):
    metrics = mocker.patch('mlflow.log_metrics')

    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    model.fit(loader.train_fn(batch_size=1), callbacks=[MLFlowCallback()])

    metrics.assert_called()
    assert metrics.call_count == 1

    metrics.reset_mock()
    model.fit(loader.train_fn(batch_size=1), callbacks=[MLFlowCallback(log_batch=True)])

    metrics.assert_called()
    assert metrics.call_count == 10

def test_MLFlowTimingCallback(mocker):
    log_metric = mocker.patch('mlflow.log_metric')

    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = MLFlowTimingCallback(batch_size=1)
    model.fit(loader.train_fn(batch_size=1), callbacks=[hook])

    log_metric.assert_called_once()

    assert log_metric.call_args[0][0] == 'train_duration'
    assert log_metric.call_args[0][1] > 0.

def test_MLFlowTimingCallback_predict(mocker):
    log_metric = mocker.patch('mlflow.log_metric')

    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = MLFlowTimingCallback(batch_size=1)
    model.predict(loader.test_fn(batch_size=1), callbacks=[hook])

    log_metric.assert_called()
    assert log_metric.call_count == 2

    args = log_metric.call_args_list[0][0]
    assert args[0] == 'test_duration'
    assert args[-1] > 0.

    args = log_metric.call_args_list[1][0]
    assert args[0] == 'test_samples_per_sec'
    assert args[-1] > 0.

def test_MLFlowTimingCallback_train_with_validation(mocker):
    log_metric = mocker.patch('mlflow.log_metric')

    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = MLFlowTimingCallback(batch_size=1)
    model.fit(loader.train_fn(batch_size=1), validation_data=loader.test_fn(batch_size=1), callbacks=[hook])

    log_metric.assert_called()
    assert log_metric.call_count == 3

    args = log_metric.call_args_list[0][0]
    assert args[0] == 'val_duration'
    assert args[-1] > 0.

    args = log_metric.call_args_list[1][0]
    assert args[0] == 'val_samples_per_sec'
    assert args[-1] > 0.

    args = log_metric.call_args_list[2][0]
    assert args[0] == 'train_duration'
    assert args[-1] > 0.

def test_MLFlowTimingCallback_multiple_epochs(mocker):
    log_metric = mocker.patch('mlflow.log_metric')

    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = MLFlowTimingCallback(batch_size=1, warmup_steps=0)
    model.fit(loader.train_fn(batch_size=1), callbacks=[hook], epochs=5)

    log_metric.assert_called()
    assert log_metric.call_count == 11

    # We expect 5 calls to epoch duration & train samples per sec
    # Followed by one call to train duration at the end
    expected_calls = zip(['epoch_duration' for i in range(5)], ['train_samples_per_sec' for i in range(5)])
    expected_calls = list(chain.from_iterable(expected_calls))
    expected_calls.append('train_duration')

    for call, expected_call in zip(log_metric.call_args_list, expected_calls):
        args, kwargs = call
        assert args[0] == expected_call
        assert args[-1] > 0
