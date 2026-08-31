from pandas import DataFrame

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def createDurationModel():
  """
  Model 1: input deck -> total duration in seconds.
  """
  return TransformedTargetRegressor(
    regressor=make_pipeline(
      TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b"
      ),
      MLPRegressor(
        hidden_layer_sizes=(8,),
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
      ),
    ),
    transformer=StandardScaler(),
  )

def createUsageModel():
  """
  Model 2: input deck + position in the run -> CPU and memory.
  """
  features = ColumnTransformer([
    (
      "deck",
      TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b"
      ),
      "deck",
    ),
    (
      "time", StandardScaler(), ["time_fraction"]
    ),
  ])

  return TransformedTargetRegressor(
    regressor=make_pipeline(
      features,
      MLPRegressor(
        hidden_layer_sizes=(16, 8),
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
      ),
    ),
    transformer=StandardScaler(),
  )

def trainModel(data: DataFrame, jobs):
  """
  Train:
  - duration_model to predict total runtime and
  - usage_model to predict CPU and memory during the run.
  """

  # Model 1: duration
  duration_model = createDurationModel()
  duration_model.fit(jobs["deck"], jobs["duration_s"])
  print("M, duration model trained")

  # Model 2: CPU and memory over time
  usage_data = data.merge(jobs[["job_id", "duration_s"]], on="job_id")

  # Normalizing time allows jobs with different durations to use the same model
  usage_data["time_fraction"] = (
    usage_data["time_rel_s"] / usage_data["duration_s"]
  ).fillna(0.0).clip(0.0, 1.0)

  usage_model = createUsageModel()
  usage_model.fit(
    usage_data[["deck", "time_fraction"]],
    usage_data[["cpu_cores_used", "Active"]]
  )
  print("M, CPU/memory usage model trained")

  # Typical interval between two monitoring measurements (in seconds).
  sample_period_s = float(
    data["time_delta_s"].dropna().median()
  )

  return {
    "duration_model": duration_model,
    "usage_model": usage_model,
    "sample_period_s": sample_period_s,
  }
