from common.loading import load_simulations
from common.Simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

MIN_PAIRED_RUNS = 10

def display_correlation() -> None:
  path: Path = Path("data")
  simulations: list[Simulation] = load_simulations(path)

  features = ["nx", "ny", "L", "nsteps", "temp", "rho", "rc"]
  columns = ["particle count", *features]

  rows = [] # Collect one dictionary per simulation
  for s in simulations:
    row = {}
    row["particle count"] = s.particlecount

    # Go through the parsed entries in its input deck
    for f in s.inputdeck:
      command = f.command

      # Keep only the variables selected above
      if command in features:
        row[command] = f.arguments

    # Add the completed row after processing the input deck
    rows.append(row)

  # Build the table in the order given by columns
  datas = pd.DataFrame(rows, columns=columns)
  # Convert values to numbers and mark invalid values as missing
  datas = datas.apply(pd.to_numeric, errors="coerce")

  # Spearman correlations using runs where both variables have values (needs at
  # least MIN_PAIRED_RUNS)
  correlations = datas.corr(method="spearman", min_periods=MIN_PAIRED_RUNS)

  # ============================================================ Figure and plot
  fig, ax = plt.subplots(figsize=(10, 6))

  ax.set_title(
    f"Spearman correlations across {len(simulations)} successful simulations"
  )

  sns.heatmap(
    correlations,
    ax=ax,
    vmin=-1, vmax=1,
    cmap="RdBu", annot=True,
    linewidths=0.4, linecolor="white"
  )

  plt.show()
