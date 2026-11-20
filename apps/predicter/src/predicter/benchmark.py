from dataclasses import dataclass
from .evaluation import evaluate_predictions, TargetScore
from .models import MODEL_FACTORIES
import numpy as np
from typing import Any

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
  results = []
  for model_name, model_factory in MODEL_FACTORIES.items():
    model = model_factory()
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    scores = evaluate_predictions(y_validation, predictions)
    result = ModelBenchmark(name=model_name, model=model, scores=scores)
    results.append(result)

  return results


def display_benchmark(results: list[ModelBenchmark]) -> None:
  baseline = next(
    (result for result in results if result.name == "dummy"),
    None,
  )

  if baseline is None:
    raise ValueError("Dummy baseline model not found.")

  baseline_by_target = {
    score.target: score
    for score in baseline.scores
  }

  def improvement(baseline_value: float, value: float) -> float:
    if baseline_value == 0:
      return 0.0 if value == 0 else float("-inf")
    return (baseline_value - value) / baseline_value * 100

  def get_status(mae_delta: float, rmse_delta: float) -> str:
    if mae_delta >= 30 and rmse_delta >= 30:
      return "EXCELLENT"
    if mae_delta >= 10 and rmse_delta >= 10:
      return "GOOD"
    if mae_delta > 0 and rmse_delta > 0:
      return "BETTER"
    if mae_delta < 0 or rmse_delta < 0:
      return "WORSE"
    return "SAME"

  targets = list(baseline_by_target)

  model_width = max(12, max(len(r.name) for r in results) + 2)
  target_width = max(10, max(len(str(t)) for t in targets) + 2)

  header = (
    f"{'MODEL':<{model_width}}"
    f"{'TARGET':<{target_width}}"
    f"{'MAE':>12}"
    f"{'RMSE':>12}"
    f"{'MAE Δ%':>12}"
    f"{'RMSE Δ%':>12}"
    f"{'STATUS':>14}"
  )

  for target in targets:
      rows = []

      for result in results:
          score = next(
              (s for s in result.scores if s.target == target),
              None,
          )

          if score is None:
              continue

          baseline_score = baseline_by_target[target]

          mae_delta = improvement(baseline_score.mae, score.mae)
          rmse_delta = improvement(baseline_score.rmse, score.rmse)

          status = (
            "BASELINE"
            if result.name == "dummy"
            else get_status(mae_delta, rmse_delta)
          )

          rows.append(
            (result.name, score, mae_delta, rmse_delta, status)
          )

      rows.sort(key=lambda row: row[1].rmse)

      non_baseline = [
        row for row in rows if row[0] != "dummy"
      ]

      best_rmse = (
        min(row[1].rmse for row in non_baseline)
        if non_baseline
        else None
      )

      print(f"\nTARGET: {target}")
      print("-" * len(header))
      print(header)
      print("-" * len(header))

      for name, score, mae_delta, rmse_delta, status in rows:
          if (
            best_rmse is not None
            and name != "dummy"
            and score.rmse == best_rmse
            and score.rmse < baseline_by_target[target].rmse
          ):
            status = "BEST ★"

          print(
            f"{name:<{model_width}}"
            f"{str(target):<{target_width}}"
            f"{score.mae:>12.4f}"
            f"{score.rmse:>12.4f}"
            f"{mae_delta:>+11.1f}%"
            f"{rmse_delta:>+11.1f}%"
            f"{status:>14}"
          )

  print("\n" + "=" * len(header))
  print("Δ% = Error reduction compared to dummy baseline")
  print("Positive Δ% = Better | Negative Δ% = Worse")
  print("BEST ★ = Lowest RMSE among models, beating baseline")
  print("=" * len(header) + "\n")
