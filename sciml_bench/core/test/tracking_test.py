from pathlib import Path
from tinydb import TinyDB, Query
from sciml_bench.core.tracking import TrackingClient

def test_create_benchmark(tmpdir):

    name = 'my-benchmark.json'
    client = TrackingClient(tmpdir / name)
    client.log_metric('loss', 1, step=1)
    client.log_metric('loss', 1, step=2)

    path = Path(tmpdir / name).with_suffix('.json')
    assert path.exists()

    with TinyDB(path) as db:
        assert db.count(Query().name == 'loss') == 2

def test_log_metrics_multi(tmpdir):
    name = 'my-benchmark.json'
    client = TrackingClient(tmpdir / name)
    client.log_metrics({'loss': 1, 'acc': .99}, step=1)

    path = Path(tmpdir / name).with_suffix('.json')
    assert path.exists()

    with TinyDB(path) as db:
        assert db.count(Query().type == 'metric') == 2
