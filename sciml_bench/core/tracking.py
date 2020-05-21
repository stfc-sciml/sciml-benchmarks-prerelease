import time
import copy
from tinydb import TinyDB, Query

def sanitize_dict(d):
    d = copy.deepcopy(d)
    for k, v in d.items():
        if type(v) is dict:
            d[k] = sanitize_dict(v)
        else:
            if isinstance(v, set):
                v = list(v)
            else:
                v = str(v)
            d[k] = v

    return d

class TrackingClient:

    def __init__(self, path):
        self._db = TinyDB(path)

    def log_metric(self, key, value, step=0):
        metric = {'name': key, 'value': value, 'step': step, 'timestamp': time.time(), 'type': 'metric'}

        metric = sanitize_dict(metric)
        self._db.insert(metric)

    def log_tag(self, key, value):
        tag = {'name': key, 'value': value, 'type': 'tag'}
        tag = sanitize_dict(tag)
        self._db.insert(tag)

    def log_param(self, key, value):
        param = {'name': key, 'value': value, 'type': 'param'}
        param = sanitize_dict(param)
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
