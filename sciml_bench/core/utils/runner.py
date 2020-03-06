from pathlib import Path

def setup_run(**params):
    num_replicas = params['num_replicas']
    params['global_batch_size'] = params['batch_size'] * num_replicas

    if params['lr_scaling'] == 'linear':
        params['learning_rate'] *= num_replicas

    Path(params['model_dir']).mkdir(parents=True, exist_ok=True)
    return params
