import time
from sciml_bench.core.test.helpers import fake_model_fn, FakeDataLoader
from sciml_bench.core.utils.hooks.mlflow import MLFlowDeviceLogger, MLFlowHostLogger, MLFlowCallback, log_device_stats, log_host_stats

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

