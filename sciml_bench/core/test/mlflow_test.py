import time
from sciml_bench.core.utils.hooks.mlflow import MLFlowDeviceLogger, MLFlowHostLogger, log_device_stats, log_host_stats

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
