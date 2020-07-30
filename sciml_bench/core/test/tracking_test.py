from pathlib import Path
from tinydb import TinyDB, Query
from sciml_bench.core.tracking import TrackingClient, sanitize_dict


def test_sanitize_dict():
    valid_dict = {'loss': set([1, 2, 3]), 'acc': .99}
    new_dict = sanitize_dict(valid_dict)
    assert new_dict == {'loss': [1, 2, 3], 'acc': .99}

    dict_with_objs = {'metrics': object()}
    new_dict = sanitize_dict(dict_with_objs)
    assert isinstance(new_dict['metrics'], str)


def test_create_benchmark(tmpdir):

    name = 'my-benchmark.json'
    client = TrackingClient(tmpdir / name)
    client.log_metric('log', {'loss': 1}, step=1)
    client.log_metric('log', {'loss': 1}, step=2)

    path = Path(tmpdir / name).with_suffix('.json')
    assert path.exists()

    with TinyDB(path) as db:
        assert db.count(Query().name == 'log') == 2


def test_log_metric(tmpdir):
    name = 'my-benchmark.json'
    client = TrackingClient(tmpdir / name)
    client.log_metric('log', {'loss': 1, 'acc': .99}, step=1)

    path = Path(tmpdir / name).with_suffix('.json')
    assert path.exists()

    with TinyDB(path) as db:
        assert db.count(Query().type == 'metric') == 1


def test_log_metrics_non_JSON_type(tmpdir):
    name = 'my-benchmark.json'
    client = TrackingClient(tmpdir / name)
    client.log_metric('log', {'loss': set([1, 2, 3]), 'acc': .99}, step=1)

    path = Path(tmpdir / name).with_suffix('.json')
    assert path.exists()

    with TinyDB(path) as db:
        assert db.count(Query().type == 'metric') == 1


def test_create_tracking_client_no_folder(tmpdir):
    name = 'my-benchmark.json'
    tmpdir = tmpdir / 'test-folder'
    TrackingClient(tmpdir / name)

    path = Path(tmpdir / name).with_suffix('.json')
    assert path.exists()
