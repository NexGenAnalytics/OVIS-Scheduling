from collections import defaultdict
import csv
from io import StringIO
import shlex

from common.models import Feature, Metric

def parse_inputdeck(content: str) -> list[Feature]:
  """
  Convert LAMMPS input deck into Feature objects.
  """
  features: list[Feature] = []

  pending = ""
  for line_number, line in enumerate(content.splitlines(), start=1):
    # LAMMPS joins lines ending in '&' before tokenizing the command.
    if line.rstrip().endswith("&"):
      pending += line.rstrip()[:-1] + " "
      continue
    line, pending = pending + line, ""
    try:
      tokens = shlex.split(line, comments=True, posix=True)
    except ValueError as error:
      raise ValueError(f"Invalid input-deck {line_number}") from error

    if not tokens: continue

    command, *arguments = tokens
    feature = Feature(command = command, arguments = arguments)
    features.append(feature)

  if pending:
    raise ValueError("Unterminated input-deck continuation")
  return features

def parse_baseline(content: str) -> dict[int, dict[str, float]]:
  """Idle value per node, then per metric."""
  baseline: dict[int, dict[str, float]] = defaultdict(dict)
  for row in csv.DictReader(StringIO(content)):
    baseline[int(row["component_id"])][row["metric"]] = float(row["value"])
  return dict(baseline)

def parse_runprofile(content: str) -> tuple[list[Metric], float]:
  metrics: list[Metric] = []
  times = []

  reader = csv.DictReader(StringIO(content))

  for line_number, row in enumerate(reader, start=2):
    try:
      metric = Metric(
        timestamp=float(row["timestamp"]),
        time_rel_s=float(row["time_rel_s"]),
        metric=row["metric"],
        value=float(row["value"]),
        component_id=int(row["component_id"]),
      )
      metrics.append(metric)
      times.append(metric.time_rel_s)
    except (TypeError, ValueError) as error:
      raise ValueError(f"Invalid {line_number}") from error

  total_time = max(times) if times else 0
  return metrics, total_time
