import json
from pathlib import Path
from tensorflow.keras.models import load_model


def main():
    path = Path('models/reset_model.h5')

    model = load_model(str(path))
    model_file = Path('models/dms_classifier/model.json')
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open('w') as handle:
        json.dump(model.to_json(), handle)


if __name__ == "__main__":
    main()
