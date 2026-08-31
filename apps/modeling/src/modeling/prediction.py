import numpy as np

from pandas import DataFrame

def predictModel(models, deck: str):
  duration_model = models["duration_model"]
  usage_model = models["usage_model"]
  sample_period_s = models["sample_period_s"]

  # Model 1
  duration_pre = duration_model.predict([deck])
  duration_s = float(duration_pre[0])
  duration_s = max(0.0, duration_s)
  times = np.arange(0.0, duration_s, sample_period_s)

  time_fraction = np.zeros_like(times)
  if duration_s > 0:
    time_fraction = times / duration_s

  prediction_data = DataFrame({
    "deck": [deck] * len(times),
    "time_fraction": time_fraction
  })

  # Model 2
  usage = usage_model.predict(prediction_data)

  # Prediction
  prediction = DataFrame({
    "time_s": times,
    "cpu_cores_used": usage[:, 0],
    "Active": usage[:, 1]
  })
  return prediction

def displayPrediction(prediction: DataFrame):
  print("M, prediction:")
  print(prediction.to_string(index=False))
