from collections import Counter
from common.loading import load_simulations
import matplotlib.pyplot as plt
from pathlib import Path

def display_xy(x: str, y: str) -> None:
  path: Path = Path("data")
  simulations: list[Simulation] = load_simulations(path)

  x_values = []
  y_values = []

  for s in simulations:
    variables = {
      feature.command: feature.arguments
      for feature in s.inputdeck
    }
    variables["particlecount"] = s.particlecount
    variables["totaltime"] = s.totaltime

    try:
      x_value = float(variables[x])
      y_value = float(variables[y])
    except (KeyError, TypeError, ValueError):
      continue

    x_values.append(x_value)
    y_values.append(y_value)

  if not x_values or not y_values:
    print(f"No simulations have numeric values for both {x!r} and {y!r}.")
    return

  point_counts = Counter(zip(x_values, y_values))
  coordinates, counts = zip(*point_counts.most_common())
  plot_x, plot_y = zip(*coordinates)

  # ============================================================ Figure and plot
  fig, ax = plt.subplots(figsize=(10, 6))

  ax.scatter(
    plot_x, plot_y, alpha=0.7,
    s=[24 * count for count in counts],
    edgecolors="white",
    linewidths=0.6,
  )

  ax.set(xscale="log", yscale="log")
  ax.set(xlabel=x, ylabel=y, title=f"{y} vs {x} ({len(simulations)} runs)")
  ax.grid(True, which="major", alpha=0.25)

  fig.tight_layout()
  plt.show()
