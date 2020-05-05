import time
import horovod.tensorflow as hvd
from itertools import chain

from sciml_bench.core.tracking import TrackingClient
from sciml_bench.core.test.helpers import fake_model_fn, FakeDataLoader
from sciml_bench.core.callbacks import DeviceLogger, HostLogger, Callback, TimingCallback, NodeLogger

def test_DeviceLogger(mocker, tmpdir):

    logger = DeviceLogger(tmpdir, interval=0.1)

    run_stub = mocker.spy(logger, 'run')

    logger.start()
    time.sleep(1)
    logger.stop()

    run_stub.assert_called()

def test_HostLogger(mocker, tmpdir):

    logger = HostLogger(tmpdir, interval=0.1)

    run_stub = mocker.spy(logger, 'run')

    logger.start()
    time.sleep(1)
    logger.stop()

    run_stub.assert_called()

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
    model.fit(loader.train_fn(batch_size=1), callbacks=[Callback(tmpdir)])

    model.fit(loader.train_fn(batch_size=1), callbacks=[Callback(tmpdir, log_batch=True)])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('loss')) == 10

def test_TimingCallback(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TimingCallback(tmpdir, batch_size=1)
    model.fit(loader.train_fn(batch_size=1), callbacks=[hook])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('train_duration')) == 1


def test_TimingCallback_predict(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TimingCallback(tmpdir, batch_size=1)
    model.predict(loader.test_fn(batch_size=1), callbacks=[hook])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('test_duration')) == 1
    assert len(db.get_metric('test_samples-per-sec')) == 1

def test_TimingCallback_train_with_validation(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TimingCallback(tmpdir, batch_size=1)
    model.fit(loader.train_fn(batch_size=1), validation_data=loader.test_fn(batch_size=1), callbacks=[hook])

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('val_duration')) == 1
    assert len(db.get_metric('val_samples-per-sec')) == 1
    assert len(db.get_metric('train_duration')) == 1

def test_TimingCallback_multiple_epochs(tmpdir):
    loader = FakeDataLoader((10, 10), (1, ))
    model = fake_model_fn((10, 10))
    hook = TimingCallback(tmpdir, batch_size=1, warmup_steps=0)
    model.fit(loader.train_fn(batch_size=1), callbacks=[hook], epochs=5)

    # We expect 5 calls to epoch duration & train samples per sec
    # Followed by one call to train duration at the end
    expected_calls = zip(['epoch_duration' for i in range(5)], ['train_samples_per_sec' for i in range(5)])
    expected_calls = list(chain.from_iterable(expected_calls))
    expected_calls.append('train_duration')

    db = TrackingClient(tmpdir / 'logs.json')
    assert len(db.get_metric('train_epoch_duration')) == 5
    assert len(db.get_metric('train_samples-per-sec')) == 5
    assert len(db.get_metric('train_duration')) == 1
