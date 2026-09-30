import argparse
import csv
from dataclasses import dataclass
import joblib
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="modeling-training")
  parser.add_argument("--simu", type=Path, required=True)
  parser.add_argument("--deta", type=int, required=True)
  parser.add_argument("--feat", nargs="+", type=str, required=True)
  return parser.parse_args()

@dataclass(frozen=True)
class Simulation:
  id: int
  inputdeck: str
  runprofile: str

def load_simulations(path: Path) -> list[Simulation]:
  simulations: list[Simulation] = []

  path_manifest = path / "ldms" / "ldms_manifest_20260921.csv"
  path_inputdecks = path / "input-decks" / "lammps"
  path_profiles = path / "ldms" / "cpu_mpi_omp"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
      status = row["status"]

      if row["status"] != "ok": continue

      job_id = row["job_id"]
      problem = row["problem"]
      filename = f"in.{problem}"

      path_inputdeck = path_inputdecks / filename
      path_profile = path_profiles / problem / job_id / "ldms_metrics.csv"

      inputdeck_content = path_inputdeck.read_text(encoding="utf-8")
      runprofile_content = path_profile.read_text(encoding="utf-8")

      simulation = Simulation(
        id = int(job_id),
        inputdeck = inputdeck_content,
        runprofile = runprofile_content,
      )
      simulations.append(simulation)

  return simulations

def create_model() -> TransformedTargetRegressor:
  regressor = make_pipeline(
    StandardScaler(),
    MLPRegressor(
      hidden_layer_sizes=(64, 32),
      solver="lbfgs",
      max_iter=2000,
      random_state=42,
    ),
  )

  return TransformedTargetRegressor(
    regressor=regressor,
    transformer=StandardScaler(),
  )

def train_cpu_model(simulations: list[Simulation], details: int, features: list[str]):
  # TODO

  return -1

def save_model(model: TransformedTargetRegressor, name: str):
  directory = Path("output/models")
  directory.mkdir(parents=True, exist_ok=True)

  filename = f"{name}_model.joblib"
  modelpath = directory / filename
  joblib.dump(model, modelpath)

  return True

def main() -> None:
  args: argparse.Namespace = init_parser()

  simulations: list[Simulation] = load_simulations(args.simu)

  cpu_model: TransformedTargetRegressor = train_cpu_model(simulations, args.deta, args.feat)

  saved: bool = save_model(cpu_model, "cpu")
  print(f"Saved?: {saved}")
