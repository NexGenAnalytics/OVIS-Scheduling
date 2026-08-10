from pandas import Index
from pathlib import Path

FIELDS = [
  "timestamp", "job_id", "build_name", "problem_name", "component_id", "Active",
  "user", "sys", "cpu_counter_delta", "time_delta_s", "cpu_cores_used",
  "time_rel_s"
]

def checkDataColums(columns: Index):
  if set(columns) != set(FIELDS):
    missing = set(FIELDS) - set(columns)
    extra = set(columns) - set(FIELDS)
    raise ValueError(
      f"CSV columns mismatch.\n"
      f"Missing: {missing}\n"
      f"Extra: {extra}"
    )
  print("M, columns checked")

def findDataCsv(path: Path):
  file_path = path / "data.csv"
  if not file_path.is_file():
    raise FileNotFoundError(
      f"The data.csv file cannot be found in:"
      f"{path}"
    )
  return file_path
