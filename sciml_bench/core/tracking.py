import time
from tinydb import TinyDB, Query

class TrackingClient:

    def __init__(self, path):
        self._db = TinyDB(path)

    def log_metric(self, key, value, step=0):
        metric = {'name': key, 'value': str(value), 'step': step, 'timestamp': time.time(), 'type': 'metric'}
        self._db.insert(metric)

    def log_metrics(self, data, step):

        metrics = []
        for key, value in data.items():
            metric = {'name': key, 'value': str(value), 'step': step, 'timestamp': time.time(), 'type': 'metric'}
            metrics.append(metric)

        self._db.insert_multiple(metrics)

    def log_tag(self, key, value):
        tag = {'name': key, 'value': str(value), 'type': 'tag'}
        self._db.insert(tag)

    def log_tags(self, data):
        tags = []
        for key, value in data.items():
            tag = {'name': key, 'value': str(value), 'timestamp': time.time(), 'type': 'tag'}
            tags.append(tag)

        self._db.insert_multiple(tags)

    def log_param(self, key, value):
        param = {'name': key, 'value': str(value), 'type': 'param'}
        self._db.insert(param)

    def log_params(self, data):
        params = []
        for key, value in data.items():
            param = {'name': key, 'value': str(value), 'timestamp': time.time(), 'type': 'param'}
            params.append(param)

        self._db.insert_multiple(params)

    def log_artifact(self, path):
        artifact = {'value': str(path), 'type': 'artifact'}
        self._db.insert(artifact)

    def get_metric(self, name):
        query = Query()
        return self._db.search((query.name == name) & (query.type == 'metric'))

    def get_param(self, name):
        query = Query()
        return self._db.search((query.name == name) & (query.type == 'param'))

    def get_tag(self, name):
        query = Query()
        return self._db.search((query.name == name) & (query.type == 'tag'))
