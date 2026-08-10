import joblib
from pathlib import Path

OUTPUT_FOLDER = Path("data/output/")
MODEL_PATH = OUTPUT_FOLDER / "model.joblib"

def saveModel(model):
  MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(model, MODEL_PATH)
  print("M, model saved")

def loadModel():
  model = joblib.load(MODEL_PATH)
  print("M, model loaded")
  return model
