import json
import sciml_bench.mark
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers.experimental.preprocessing import Normalization


@sciml_bench.mark.model_function('dms_classifier')
def resnet_classifier(input_shape, **kwargs):
    with open('models/dms_classifier/model.json') as handle:
        config = json.load(handle)
    model = model_from_json(config, custom_objects={'Normalization': Normalization})
    return model
