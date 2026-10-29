import numpy as np

from .evaluation import evaluate_predictions
from .models import MODEL_FACTORIES

def run_benchmark(
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_validation: np.ndarray,
  y_validation: np.ndarray,
):
  for model_name, model_factory in MODEL_FACTORIES.items():
    model = model_factory()
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    scores = evaluate_predictions(y_validation, predictions)
    print(f"TODO => {model_name}; score: {scores}")
