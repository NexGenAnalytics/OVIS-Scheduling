import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

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

      path_inputdeck = path_inputdecks / f"in.{problem}"
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

def main() -> None:
  args: argparse.Namespace = init_parser()

  simulations: list[Simulation] = load_simulations(args.simu)

  print(len(simulations))

  # print(path_profiles)
  print(args.feat)
