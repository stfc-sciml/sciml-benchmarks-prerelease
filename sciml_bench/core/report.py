
import collections
from collections import defaultdict
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, layout
from bokeh.models import Div, CustomJS, Select
from bokeh.models import ColumnDataSource, DataTable, TableColumn

from sciml_bench.core.tracking import TrackingClient


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_metrics_explorer(client):
    metrics = client.get_metrics()

    all_metrics = defaultdict(list)
    for metric in metrics:
        key = metric['name'].replace('_log', '')
        flat_metric = flatten(metric['data'])

        exec_mode = flat_metric.pop('execution_mode', '')
        exec_mode = exec_mode + '_' if exec_mode != '' else exec_mode
        flat_metric.pop('name', None)

        flat_metric = {exec_mode + key + '_' + k: v for k, v in flat_metric.items()}
        for k, v in flat_metric.items():
            all_metrics[k].append(v)

    metrics = {k: dict(values=v, steps=list(range(len(v)))) for k, v in all_metrics.items() if len(v) > 1}
    single_metrics = {k:v for k, v in all_metrics.items() if len(v) == 1}

    selection_names = sorted(list(metrics.keys()))
    current_source = ColumnDataSource(metrics[selection_names[0]])
    p = figure(plot_width=800, plot_height=400)
    p.line('steps', 'values', line_width=2, source=current_source)

    callback = CustomJS(args=dict(current_source=current_source, metrics=metrics), code="""
        var f = cb_obj.value;
        var data = metrics[f];

        current_source.data['steps'] = data['steps'];
        current_source.data['values'] = data['values'];

        current_source.change.emit();
        """)

    select = Select(title="Metrics:", value=selection_names[0],
                                       options=sorted(list(metrics.keys())))
    select.js_on_change('value', callback)

    if len(single_metrics) > 0:
        src = ColumnDataSource(dict(names=list(single_metrics.keys()), values=list(single_metrics.values())))
        columns = [
                TableColumn(field="names", title="Name"),
                TableColumn(field="values", title="Value"),
            ]
        metrics_table = DataTable(source=src, columns=columns, width=800, height=280)

        return column(select, p, metrics_table)
    else:
        return column(select, p)

def create_table(data):
    names = []
    values = []
    data = flatten(data)

    exec_mode = data.pop('execution_mode', '')
    exec_mode = exec_mode + '_' if exec_mode != '' else exec_mode
    data.pop('name', None)

    for key, value in data.items():
        names.append(exec_mode + key)
        values.append(value)


    data = dict(
	    names=names,
	    values=values,
	)

    source = ColumnDataSource(data)

    columns = [
	    TableColumn(field="names", title="Name"),
	    TableColumn(field="values", title="Value"),
	]

    return DataTable(source=source, columns=columns, width=800, height=280)


def create_report(folder):
    client = TrackingClient(folder / 'logs.json')
    params = client.get_params()[0]
    params = params['data']
    # output to static HTML file
    output_file(folder / "report.html")


    widgets = []

    param_header = Div(text="""
    <h2> Model Parameters </h2>
    """)
    widgets.append(param_header)

    param_table = create_table(params)
    widgets.append(param_table)

    widgets.append(Div(text="""
    <h2> Model Metrics </h2>
    """))
    metric_explorer = create_metrics_explorer(client)
    widgets.append(metric_explorer)

    host_files = list(folder.glob('*_host.json'))
    host_logs = {f.name.split('_')[0]: f for f in host_files}

    device_files = list(folder.glob('*_devices.json'))
    device_logs = {f.name.split('_')[0]: f for f in device_files}

    node_names = [f.name.split('_')[0] for f in host_files]

    node_logs = defaultdict(dict)
    for key in node_names:
        node_logs[key]['host'] = host_logs[key]
        node_logs[key]['devices'] = device_logs[key]

    for node_name, logs in node_logs.items():
        widgets.append(Div(text="<h2>{} Host</h2>".format(node_name)))
        client = TrackingClient(logs['host'])
        tags = client.get_tags()[0]

        tags_table = create_table(tags['data'])
        widgets.append(tags_table)

        host_explorer = create_metrics_explorer(client)
        widgets.append(host_explorer)

        widgets.append(Div(text="<h2>{} Devices</h2>".format(node_name)))

        client = TrackingClient(logs['devices'])
        tags = client.get_tags()[0]

        tags_table = create_table(tags['data'])
        widgets.append(tags_table)

        devices_explorer = create_metrics_explorer(client)
        widgets.append(devices_explorer)

    show(layout(widgets))
