"""Open an LDMS CSV as two linked Matplotlib resource-profile plots."""

import argparse
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import fmean

CPU_METRIC = "cpu_cores_used"
MEMORY_METRIC = "mem_active_kb"
METRICS = (CPU_METRIC, MEMORY_METRIC)
TIME_UNITS = {"seconds": (1.0, "s"), "minutes": (60.0, "min"), "hours": (3600.0, "h")}
MEMORY_UNITS = {"KiB": 1.0, "MiB": 1024.0, "GiB": 1024.0 ** 2}
SeriesKey = tuple[str, str]


@dataclass
class Profile:
  source: Path
  # Keep separate jobs/components separate, rather than joining unrelated points.
  series: dict[str, dict[SeriesKey, list[tuple[float, float]]]] = field(
    default_factory=lambda: {metric: {} for metric in METRICS}
  )


def load_profile(path: Path) -> Profile:
  """Read supported metrics and sort each series by its actual elapsed time."""
  profile = Profile(source=path)
  with path.open(newline="", encoding="utf-8-sig") as stream:
    reader = csv.DictReader(stream)
    missing = {"time_rel_s", "metric", "value"} - set(reader.fieldnames or [])
    if missing:
      raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    for row in reader:
      metric = row.get("metric")
      if metric not in METRICS:
        continue
      try:
        elapsed, value = float(row["time_rel_s"]), float(row["value"])
        if not all(math.isfinite(number) and number >= 0 for number in (elapsed, value)):
          raise ValueError("time and resource values must be finite and nonnegative")
        unit = (row.get("unit") or "").strip().lower()
        allowed_units = {"", "cores", "core"} if metric == CPU_METRIC else {"", "kb", "kib"}
        if unit not in allowed_units:
          raise ValueError(f"Unexpected unit {unit!r} for {metric}")
      except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {metric} sample at CSV line {reader.line_num}: {error}") from error
      key = (row.get("job_id") or "unknown", row.get("component_id") or "unknown")
      profile.series[metric].setdefault(key, []).append((elapsed, value))
  if not any(profile.series.values()):
    raise ValueError(f"No {CPU_METRIC} or {MEMORY_METRIC} samples found")
  for groups in profile.series.values():
    for samples in groups.values():
      samples.sort(key=lambda sample: sample[0])
  return profile


def plot_profile(profile: Profile, *, time_unit: str = "minutes", memory_unit: str = "MiB"):
  """Return a figure with CPU above memory; toolbar zoom/pan shares the time axis."""
  import matplotlib.pyplot as plt

  time_scale, time_label = TIME_UNITS[time_unit]
  figure, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7), layout="constrained")
  figure.canvas.manager.set_window_title(f"LDMS profiles — {profile.source.name}")
  figure.suptitle(f"CPU and memory profiles\n{profile.source.name}", fontsize=14)
  keys = sorted({key for groups in profile.series.values() for key in groups})
  palette = plt.get_cmap("tab10")
  colors = {key: palette(index % 10) for index, key in enumerate(keys)}
  definitions = (
    (CPU_METRIC, "CPU usage", "CPU cores used", 1.0, "cores", "#148578"),
    (MEMORY_METRIC, "Active memory", f"Active memory ({memory_unit})",
     MEMORY_UNITS[memory_unit], memory_unit, "#3669ad"),
  )
  for axis, (metric, title, ylabel, scale, value_unit, single_color) in zip(axes, definitions):
    groups = profile.series[metric]
    count = sum(len(samples) for samples in groups.values())
    axis.set_title(f"{title} · {count:,} samples", loc="left", fontsize=11, pad=10)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.margins(x=0.02, y=0.15)
    for key, samples in sorted(groups.items()):
      times = [elapsed / time_scale for elapsed, _ in samples]
      values = [value / scale for _, value in samples]
      color = single_color if len(keys) == 1 else colors[key]
      label = f"Job {key[0]} / component {key[1]}"
      axis.plot(times, values, color=color, linewidth=1.6,
                marker="." if len(samples) <= 100 else None, markersize=4, label=label)
      if len(groups) == 1:
        axis.fill_between(times, values, alpha=0.08, color=color)
        summary = (f"Min {min(values):,.2f}  ·  Sample mean {fmean(values):,.2f}"
                   f"  ·  Max {max(values):,.2f} {value_unit}")
        axis.text(0.99, 0.97, summary, ha="right", va="top", transform=axis.transAxes,
                  fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    if not groups:
      axis.text(0.5, 0.5, f"No {metric} samples in this file", ha="center", va="center",
                transform=axis.transAxes, color="#666666")
    else:
      axis.legend(loc="lower left", fontsize=8, framealpha=0.9)
    axis.set_ylim(bottom=0)
  axes[-1].set_xlabel(f"Elapsed time ({time_label})")
  return figure


def choose_csv() -> Path | None:
  """Use a native file picker when the command is run without a CSV path."""
  try:
    import tkinter as tk
    from tkinter import filedialog
  except ImportError as error:
    raise RuntimeError("File picker unavailable. Pass the CSV path on the command line.") from error
  try:
    root = tk.Tk()
  except tk.TclError as error:
    raise RuntimeError("File picker unavailable. Pass the CSV path on the command line.") from error
  try:
    root.withdraw()
    filename = filedialog.askopenfilename(
      parent=root, title="Choose an LDMS metrics CSV",
      filetypes=[("CSV files", "*.csv"), ("All files", "*")],
    )
    return Path(filename) if filename else None
  finally:
    root.destroy()


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="visualizer", description=__doc__)
  parser.add_argument("csv", nargs="?", type=Path, help="Path to ldms_metrics.csv; omit to use a file picker.")
  parser.add_argument("--time-unit", choices=TIME_UNITS, default="minutes")
  parser.add_argument("--memory-unit", choices=MEMORY_UNITS, default="MiB")
  parser.add_argument("--save", type=Path, help="Also save the two plots as PNG, PDF or SVG.")
  parser.add_argument("--no-show", action="store_true", help="Save without opening a window (requires --save).")
  args = parser.parse_args(argv)
  if args.no_show and args.save is None:
    parser.error("--no-show requires --save")
  if args.no_show and args.csv is None:
    parser.error("--no-show requires a CSV path")
  try:
    path = args.csv if args.csv is not None else choose_csv()
    if path is None:
      return 0  # File-picker cancellation.
    profile = load_profile(path)
    import matplotlib
    if args.no_show:
      matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plot_profile(profile, time_unit=args.time_unit, memory_unit=args.memory_unit)
    if args.save:
      args.save.parent.mkdir(parents=True, exist_ok=True)
      figure.savefig(args.save, dpi=180)
      print(f"Saved profiles to {args.save}")
    if args.no_show:
      plt.close(figure)
    else:
      plt.show()
  except ModuleNotFoundError as error:
    parser.exit(2, f"{error}. Install the app: python -m pip install -e apps/visualizer\n")
  except (OSError, ValueError, RuntimeError, csv.Error) as error:
    parser.exit(2, f"Unable to display profiles: {error}\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
