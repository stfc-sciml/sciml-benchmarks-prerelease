from pathlib import Path
import time
import numpy as np
from tinydb import TinyDB, Query


def sanitize_dict(d):
    d = d.copy()
    for k, v in d.items():
        if type(v) is dict:
            v = sanitize_dict(v)
        elif isinstance(v, np.floating) or isinstance(v, float):
            v = float(v)
        elif isinstance(v, set):
            v = list(v)
        elif hasattr(v, '__name__'):
            v = v.__name__
        else:
            v = str(v)
        d[k] = v
    return d


class TrackingClient:

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = TinyDB(path)

    def log_metric(self, key, value, step=0):
        value = sanitize_dict(value)
        metric = {'name': key, 'data': value, 'step': step,
                  'timestamp': time.time(), 'type': 'metric'}

        self._db.insert(metric)

    def log_tag(self, key, value):
        value = sanitize_dict(value)
        tag = {'name': key, 'data': value, 'type': 'tag'}
        self._db.insert(tag)

    def log_param(self, key, value):
        value = sanitize_dict(value)
        param = {'name': key, 'data': value, 'type': 'param'}
        self._db.insert(param)

    def get_metric(self, name):
        query = Query()
        return self._db.search((query.name == name) & (query.type == 'metric'))

    def get_metrics(self):
        query = Query()
        return self._db.search(query.type == 'metric')

    def get_param(self, name):
        query = Query()
        return self._db.search((query.name == name) & (query.type == 'param'))

    def get_params(self):
        query = Query()
        return self._db.search(query.type == 'param')

    def get_tag(self, name):
        query = Query()
        return self._db.search((query.name == name) & (query.type == 'tag'))

    def get_tags(self):
        query = Query()
        return self._db.search(query.type == 'tag')
