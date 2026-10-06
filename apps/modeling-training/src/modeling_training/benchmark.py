from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluation import TargetScore, evaluate_predictions
from .models import MODEL_FACTORIES

@dataclass
class ModelBenchmark:
  name: str
  model: Any
  scores: list[TargetScore]

def run_benchmark(
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_validation: np.ndarray,
  y_validation: np.ndarray,
) -> list[ModelBenchmark]:
  results: list[ModelBenchmark] = []

  for model_name, model_factory in MODEL_FACTORIES.items():
    model = model_factory()
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    scores = evaluate_predictions(y_validation, predictions)

    results.append(ModelBenchmark(
      name=model_name,
      model=model,
      scores=scores,
    ))

  return results

def display_benchmark(
  results: list[ModelBenchmark],
) -> None:
  baseline = next(
    result for result in results
    if result.name == "dummy"
  )

  baseline_by_target = {
    score.target: score
    for score in baseline.scores
  }

  print(
    f"{'MODEL':<10}"
    f"{'TARGET':<10}"
    f"{'MAE':>14}"
    f"{'RMSE':>14}"
    f"{'MAE VS DUMMY':>14}"
    f"{'RMSE VS DUMMY':>17}"
  )

  for result in results:
    for score in result.scores:
      baseline_mae = baseline_by_target[score.target].mae
      baseline_rmse = baseline_by_target[score.target].rmse

      mae_improvement = (
        (baseline_mae - score.mae) / baseline_mae * 100
        if baseline_mae != 0
        else 0.0
      )

      rmse_improvement = (
        (baseline_rmse - score.rmse) / baseline_rmse * 100
        if baseline_rmse != 0
        else 0.0
      )

      print(
        f"{result.name:<10}"
        f"{score.target:<10}"
        f"{score.mae:>14.4f}"
        f"{score.rmse:>14.4f}"
        f"{mae_improvement:>13.1f}%"
        f"{rmse_improvement:>13.1f}%"
      )

def select_best_model(
  results: list[ModelBenchmark],
  target: str = "mean",
) -> ModelBenchmark:
  def target_mae(result: ModelBenchmark) -> float:
    for score in result.scores:
      if score.target == target:
        return score.mae

    raise ValueError(f"Target {target!r} was not evaluated")

  return min(results, key=target_mae)
