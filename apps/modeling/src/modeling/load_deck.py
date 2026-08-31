from pathlib import Path
from pandas import DataFrame
from common.utils import read_file

def loadDeck(path: Path) -> str:
  if not path.is_file():
    raise FileNotFoundError(
      f"Input deck not found: {path}"
    )
  content = read_file(path, str)
  return "\n".join(content)

def loadDecks(data: DataFrame, folder_path: Path):
  decks = {}
  jobs = data["job_id"].unique()
  for job_id in jobs:
    deck_path = folder_path / f"{int(job_id)}.txt"
    decks[job_id] = loadDeck(deck_path)
  data["deck"] = data["job_id"].map(decks)
  jobs = checkDataframe(data)
  return data, jobs

def checkDataframe(data: DataFrame):
  # One training row per job for the duration model.
  jobs = (
    data.groupby("job_id", as_index=False)
    .agg(
      deck=("deck", "first"),
      duration_s=("time_rel_s", "max"),
    )
  )

  if len(jobs) < 2:
    raise ValueError(
      "At least two jobs are required to train the models"
    )

  return jobs
