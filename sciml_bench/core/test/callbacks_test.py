import time
import horovod.tensorflow as hvd
from itertools import chain

from sciml_bench.core.tracking import TrackingClient
from sciml_bench.core.test.helpers import fake_model_fn, FakeDataLoader
from sciml_bench.core.callbacks import DeviceLogger, HostLogger, TrackingCallback, NodeLogger

def test_DeviceLogger(mocker, tmpdir):

    logger = DeviceLogger(tmpdir, interval=0.1)

    logger.start()
    time.sleep(1)
    logger.stop()

    assert (tmpdir / 'devices.json').exists()

    db = TrackingClient(tmpdir / 'devices.json')
    logs = db.get_metric('device_log')
    assert len(logs) > 1

    if logs[0] == {}:
        return

    for log in logs:
        log = log['data']
        gpu = log['gpu_0']
        assert 'execution_mode' in log
        assert 'name' in log
        assert 'utilization' in gpu
        assert 'power' in gpu
        assert 'memory' in gpu
        assert 'free' in gpu['memory']
        assert 'used' in gpu['memory']

def test_HostLogger(mocker, tmpdir):

    logger = HostLogger(tmpdir, interval=0.1)

    logger.start()
    time.sleep(1)
    logger.stop()

    assert (tmpdir / 'host.json').exists()

    db = TrackingClient(tmpdir / 'host.json')
    logs = db.get_metric('host_log')
    assert len(logs) > 1

    for log in logs:
        log = log['data']
        assert 'execution_mode' in log
        assert 'name' in log
        assert 'cpu' in log
        assert 'percent' in log['cpu']
        assert 'memory' in log
        assert 'free' in log['memory']
        assert 'used' in log['memory']
        assert 'available' in log['memory']
        assert 'utilization' in log['memory']
        assert 'disk' in log
        assert 'net' in log


def test_NodeLogger(mocker, tmpdir):
    hvd.init()
    logger = NodeLogger(tmpdir, 'pytest', interval=0.1)

    host_stub = mocker.spy(logger._host_logger, 'run')
    device_stub = mocker.spy(logger._device_logger, 'run')

    with logger:
        time.sleep(1)

    host_stub.assert_called()
    device_stub.assert_called()


def test_Callback(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))

    model.fit(loader.train_fn(batch_size=1), callbacks=[TrackingCallback(tmpdir, batch_size=1, log_batch=True)])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('train_log')) == 1

def test_TrackingCallback(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TrackingCallback(tmpdir, batch_size=1)
    model.fit(loader.train_fn(batch_size=1), callbacks=[hook])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('train_log')) == 1


def test_TrackingCallback_predict(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TrackingCallback(tmpdir, batch_size=1)
    model.predict(loader.test_fn(batch_size=1), callbacks=[hook])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('predict_log')) == 1

def test_TrackingCallback_train_with_validation(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TrackingCallback(tmpdir, batch_size=1)
    model.fit(loader.train_fn(batch_size=1), validation_data=loader.test_fn(batch_size=1), callbacks=[hook])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('test_log')) == 1

def test_TrackingCallback_multiple_epochs(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TrackingCallback(tmpdir, batch_size=1, warmup_steps=0)
    model.fit(loader.train_fn(batch_size=1), callbacks=[hook], epochs=5)

    # We expect 5 calls to epoch duration & train samples per sec
    # Followed by one call to train duration at the end
    expected_calls = zip(['epoch_duration' for i in range(5)], ['train_samples_per_sec' for i in range(5)])
    expected_calls = list(chain.from_iterable(expected_calls))
    expected_calls.append('train_duration')

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('epoch_log')) == 5
    assert len(db.get_metric('train_log')) == 1

    epoch_logs = db.get_metric('epoch_log')
    assert isinstance(epoch_logs, list)

    for log in epoch_logs:
        log = log['data']
        assert 'duration' in log
        assert 'loss' in log
        assert 'samples_per_sec' in log

    train_log = db.get_metric('train_log')[0]
    log = train_log['data']
    assert 'duration' in log
    assert 'samples_per_sec' in log
