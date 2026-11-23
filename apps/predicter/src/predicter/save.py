from .benchmark import ModelBenchmark
import joblib
from pathlib import Path

def save_model(result: ModelBenchmark, columns: list[str], resource: str) -> Path:
  directory = Path("outputs/models")
  directory.mkdir(parents=True, exist_ok=True)

  model_path = directory / f"{resource}.joblib"

  joblib.dump(
    {
      "model": result.model,
      "model_name": result.name,
      "columns": columns,
      "resource": resource,
      "unit": "kB" if resource == "memory" else "cores",
    },
    model_path,
  )

  return model_path
