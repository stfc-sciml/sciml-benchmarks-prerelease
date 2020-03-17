import time
from sciml_bench.core.utils.hooks.mlflow import MLFlowDeviceLogger, MLFlowHostLogger

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
