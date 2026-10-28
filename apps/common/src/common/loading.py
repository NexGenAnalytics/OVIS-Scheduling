import csv
from pathlib import Path

from common.converters import Converter, get_converter
from common.models import Simulation
from common.parsing import parse_runprofile

# Reference configuration: nodes, ntasks, ntasks_per_node, omp_num_threads.
REFERENCE_CONFIGURATION = (1, 48, 48, 1)

def _is_reference(row: dict[str, str]) -> bool:
  keys = ("nodes", "ntasks", "ntasks_per_node", "omp_num_threads")
  return tuple(int(row[key]) for key in keys) == REFERENCE_CONFIGURATION

def load_simulations(path: Path, converter: Converter | None = None) -> list[Simulation]:
  """Load usable runs of the reference configuration from a data directory."""
  converter = converter or get_converter("lammps")
  simulations: list[Simulation] = []

  path_manifest = path / "ldms" / "summary_global.csv"
  path_inputdecks = path / "input-decks" / converter.name
  path_profiles = path / "ldms" / "cpu_mpi_omp"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    rows = list(csv.DictReader(csv_file))
  print(f"- Total {len(rows)} files")
  print(f"- Reference configuration (nodes, ntasks, ntasks_per_node, omp_num_threads): {REFERENCE_CONFIGURATION}")

  for row in rows:
    if row["status"] != "ok" or not _is_reference(row): continue

    job_id = row["job_id"]
    problem = row["problem"]
    inputdeck = converter.parse((path_inputdecks / f"in.{problem}").read_text(encoding="utf-8"))
    profile_csv = (path_profiles / problem / job_id / "ldms_metrics.csv").read_text(encoding="utf-8")
    runprofile, total_time = parse_runprofile(profile_csv)

    simulations.append(Simulation(
      id = int(job_id),
      inputdeck = inputdeck,
      runprofile = runprofile,
      totaltime = int(total_time),
      particlecount = float(row["particle_count"]),
      name = problem,
    ))

  print(f"- Load {len(simulations)} simulations")
  return simulations
