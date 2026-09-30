from common.models import Feature, Metric
import csv
from io import StringIO
import shlex

def parse_inputdeck(content: str) -> list[Feature]:
  """
  Convert LAMMPS input deck into Feature objects.
  """
  features: list[Feature] = []

  for line_number, line in enumerate(content.splitlines(), start=1):
    try:
      tokens = shlex.split(line, comments=True, posix=True)
    except ValueError as error:
      raise ValueError(f"Invalid input-deck {line_number}") from error

    if not tokens: continue

    command, *arguments = tokens
    feature = Feature(command = command, arguments = arguments)
    features.append(feature)

  return features

def parse_runprofile(content: str) -> list[Metric]:
  metrics: list[Metric] = []

  reader = csv.DictReader(StringIO(content))

  for line_number, row in enumerate(reader, start=2):
    try:
      metric = Metric(
        timestamp=float(row["timestamp"]),
        time_rel_s=float(row["time_rel_s"]),
        # job_id=int(row["job_id"]),
        # component_id=int(row["component_id"]),
        metric=row["metric"],
        # unit=row["unit"],
        value=float(row["value"]),
      )
      metrics.append(metric)
    except (TypeError, ValueError) as error:
      raise ValueError(f"Invalid {line_number}") from error

  return metrics
