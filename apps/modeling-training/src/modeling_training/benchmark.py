from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluation import TARGET_NAMES, TargetScore, evaluate_predictions
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
  model_width = max(10, max(len(result.name) for result in results) + 2)

  print(
    f"{'MODEL':<{model_width}}"
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
        else (0.0 if score.mae == 0 else float("-inf"))
      )

      rmse_improvement = (
        (baseline_rmse - score.rmse) / baseline_rmse * 100
        if baseline_rmse != 0
        else (0.0 if score.rmse == 0 else float("-inf"))
      )

      print(
        f"{result.name:<{model_width}}"
        f"{score.target:<10}"
        f"{score.mae:>14.4f}"
        f"{score.rmse:>14.4f}"
        f"{mae_improvement:>13.1f}%"
        f"{rmse_improvement:>13.1f}%"
      )

  print("\nCombined validation score (all targets, MAE + RMSE; lower is better):")
  for result, score in compute_selection_scores(results):
    print(f"{result.name:<{model_width}}{score:>14.4f}")

def compute_selection_scores(
  results: list[ModelBenchmark],
  target: str | None = None,
) -> list[tuple[ModelBenchmark, float]]:
  """Average errors relative to Dummy, equally weighting targets and MAE/RMSE.

  A score below 1 beats Dummy overall when its reference errors are nonzero.
  If Dummy has zero error, matching it contributes 0; a positive error
  contributes infinity, so a model cannot trade away a perfect baseline target.
  """
  if not results:
    raise ValueError("No benchmark results to select from")
  if target is not None and target not in TARGET_NAMES:
    raise ValueError(f"Unknown target {target!r}")

  baseline = next((result for result in results if result.name == "dummy"), None)
  if baseline is None:
    raise ValueError("Dummy benchmark is required to normalize selection scores")

  targets = TARGET_NAMES if target is None else (target,)

  def target_scores(result: ModelBenchmark) -> dict[str, TargetScore]:
    by_target = {score.target: score for score in result.scores}
    for name in targets:
      if name not in by_target:
        raise ValueError(f"Model {result.name!r}: target {name!r} was not evaluated")
      score = by_target[name]
      if any(not np.isfinite(value) or value < 0 for value in (score.mae, score.rmse)):
        raise ValueError(f"Model {result.name!r}: invalid errors for target {name!r}")
    return by_target

  baseline_by_target = target_scores(baseline)
  selection_scores: list[tuple[ModelBenchmark, float]] = []
  for result in results:
    by_target = target_scores(result)
    relative_errors: list[float] = []
    for name in targets:
      for metric in ("mae", "rmse"):
        error = getattr(by_target[name], metric)
        reference = getattr(baseline_by_target[name], metric)
        if reference == 0:
          relative_errors.append(0.0 if error == 0 else float("inf"))
        else:
          relative_errors.append(error / reference)
    selection_scores.append((result, sum(relative_errors) / len(relative_errors)))

  return selection_scores

def select_best_model(
  results: list[ModelBenchmark],
  target: str | None = None,
) -> ModelBenchmark:
  """Select the lowest combined validation score; ties keep benchmark order.

  By default all three targets contribute. An explicit target restricts the
  comparison to that target, still considering both MAE and RMSE.
  """
  return min(compute_selection_scores(results, target), key=lambda item: item[1])[0]
