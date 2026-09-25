import csv
import pandas as pd

from dataclasses import dataclass
from hashing.utils import create_hash
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class JobSummary:
  id: int
  build: str
  problem: str
  rows: int
  status: str

@dataclass(frozen=True)
class Simulation:
  id: int
  pathinputdeck: str
  pathrunprofile: str

@dataclass(frozen=True)
class SimulationContent:
  id: int
  inputdeckhash: int
  cpuprofile: pd.DataFrame
  memoryprofile: pd.DataFrame

def read_job_summaries(path: Path) -> List[JobSummary]:
  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)

    return [
      JobSummary(
        id=int(row["job_id"]),
        build=row["build"], problem=row["problem"],
        rows=int(row["rows"]), status=row["status"],
      )
      for row in reader
    ]

def get_valid_jobs(jobs: List[JobSummary]) -> List[Simulation]:
  # Job is "valid" if files exists
  valid_jobs = []

  for job in jobs:
    problem = job.problem

    path_profiles = (
      Path("data/ldms") / job.build / job.problem / str(job.id)
      / "ldms_metrics.csv"
    )
    path_input_deck = (Path("data/input-decks/lammps") / f"in.{job.problem}")

    if path_profiles.is_file() and path_input_deck.is_file():
      valid_jobs.append(
        Simulation(
          id=job.id,
          pathinputdeck=path_input_deck,
          pathrunprofile=path_profiles,
        )
      )
    else:
      print(f"Job {job.id}: required files are missing")

  return valid_jobs

def load_jobs(jobs: List[Simulation]) -> List[SimulationContent]:
  loaded_jobs = []

  for job in jobs:
    path_input_deck = Path(job.pathinputdeck)
    path_run_profile = Path(job.pathrunprofile)

    _, input_deck_hash = create_hash(
      str(path_input_deck),
      "lammps_in_files",
    )

    metrics = pd.read_csv(
      path_run_profile,
      dtype={
        "job_id": "int64",
        "component_id": "int64",
        "metric": "string",
        "unit": "string",
      },
    )

    required_columns = {
      "timestamp",
      "time_rel_s",
      "job_id",
      "component_id",
      "metric",
      "unit",
      "value",
    }
    missing_columns = required_columns.difference(metrics.columns)

    if missing_columns:
      raise ValueError(
        f"{path_run_profile}: missing columns "
        f"{sorted(missing_columns)}"
      )

    cpu_profile = (
      metrics.loc[metrics["metric"] == "cpu_cores_used"]
      .sort_values(["component_id", "time_rel_s"])
      .reset_index(drop=True)
    )

    memory_profile = (
      metrics.loc[metrics["metric"] == "mem_active_kb"]
      .sort_values(["component_id", "time_rel_s"])
      .reset_index(drop=True)
    )

    if cpu_profile.empty:
      raise ValueError(f"{path_run_profile}: no CPU profile was found")

    if memory_profile.empty:
      raise ValueError(f"{path_run_profile}: no memory profile was found")

    loaded_jobs.append(
      SimulationContent(
        id=job.id,
        inputdeckhash=input_deck_hash,
        cpuprofile=cpu_profile,
        memoryprofile=memory_profile,
      )
    )

  return loaded_jobs

def main() -> None:
  path_manifest = "data/ldms/ldms_manifest_20260921.csv"

  # LOADING MANIFEST
  raw_jobs: List[JobSummary] = read_job_summaries(Path(path_manifest))
  ok_jobs: List[JobSummary] = [job for job in raw_jobs if job.status == "ok"]

  # VALIDATING MANIFEST
  valid_jobs: List[Simulation] = get_valid_jobs(ok_jobs)

  # LOADING DATA
  datas: List[SimulationContent] = load_jobs(valid_jobs)

  # TRAINING MODELS
  # TODO, future usage:
  # cpu_model, memory_model = trainModels(??, ??, ??)

  # SAVE MODELS
  # TODO

