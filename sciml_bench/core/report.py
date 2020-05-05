
from collections import defaultdict
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, layout
from bokeh.models import Div, CustomJS, Select
from bokeh.models import ColumnDataSource, DataTable, TableColumn

from sciml_bench.core.tracking import TrackingClient

def create_metrics_explorer(client):
    metrics = client.get_metrics()
    names = list({m['name'] for m in metrics})

    metrics = {}
    single_metrics = {}
    for name in names:
        metric = client.get_metric(name)
        steps = [m['step'] for m in metric]
        values = [float(m['value']) for m in metric]
        if len(values) <= 1:
            single_metrics[name] = values[0]
        else:
            metrics[name] = dict(steps=steps, values=values)

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

    names = [p['name'] for p in data]
    values = [p['value'] for p in data]

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
    params = client.get_params()
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


    host_files = list(folder.glob('node_*_host.json'))
    host_logs = {f.name.split('_')[1]: f for f in host_files}

    device_files = list(folder.glob('node_*_devices.json'))
    device_logs = {f.name.split('_')[1]: f for f in device_files}

    node_names = [f.name.split('_')[1] for f in host_files]

    node_logs = defaultdict(dict)
    for key in node_names:
        node_logs[key]['host'] = host_logs[key]
        node_logs[key]['devices'] = device_logs[key]

    for node_name, logs in node_logs.items():
        widgets.append(Div(text="<h2>{} Host</h2>".format(node_name)))
        client = TrackingClient(logs['host'])
        tags = client.get_tags()

        tags_table = create_table(tags)
        widgets.append(tags_table)

        host_explorer = create_metrics_explorer(client)
        widgets.append(host_explorer)

        widgets.append(Div(text="<h2>{} Devices</h2>".format(node_name)))

        client = TrackingClient(logs['devices'])
        tags = client.get_tags()

        tags_table = create_table(tags)
        widgets.append(tags_table)

        devices_explorer = create_metrics_explorer(client)
        widgets.append(devices_explorer)

    show(layout(widgets))
