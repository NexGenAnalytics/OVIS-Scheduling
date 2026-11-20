from common.metrics import RESOURCE_METRICS
from common.parsing import parse_inputdeck
from .decker import define_deck
import joblib
import numpy as np
from pathlib import Path

def define_predict(path: str):

  for resource, metric in RESOURCE_METRICS.items():
    directory = Path("outputs/models")
    model_path = directory / f"{resource}.joblib"
    saved = joblib.load(model_path)

    inputdeck = parse_inputdeck(path)
    features = define_deck(inputdeck, saved["columns"])

    X = np.asarray([features], dtype=float)
    prediction = float(saved["model"].predict(X)[0])

    unit = "kB" if resource == "memory" else "cores"
    print(f"{prediction} {unit}")
