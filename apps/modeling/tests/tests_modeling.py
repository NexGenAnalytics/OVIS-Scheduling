import tempfile
import unittest

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

from modeling.load_deck import checkDataframe, loadDeck, loadDecks
from modeling.prediction import displayPrediction, predictModel
from modeling.training import trainModel


class TestModeling(unittest.TestCase):

  def createData(self) -> pd.DataFrame:
    return pd.DataFrame([
      {
        "job_id": 1,
        "time_rel_s": 0.0,
        "time_delta_s": None,
        "cpu_cores_used": 0.0,
        "Active": 100.0,
      },
      {
        "job_id": 1,
        "time_rel_s": 60.0,
        "time_delta_s": 60.0,
        "cpu_cores_used": 2.0,
        "Active": 110.0,
      },
      {
        "job_id": 2,
        "time_rel_s": 0.0,
        "time_delta_s": None,
        "cpu_cores_used": 0.0,
        "Active": 90.0,
      },
      {
        "job_id": 2,
        "time_rel_s": 60.0,
        "time_delta_s": 60.0,
        "cpu_cores_used": 3.0,
        "Active": 105.0,
      },
      {
        "job_id": 2,
        "time_rel_s": 120.0,
        "time_delta_s": 60.0,
        "cpu_cores_used": 4.0,
        "Active": 120.0,
      },
    ])

  def createDeckFiles(self, folder: Path) -> None:
    (folder / "1.txt").write_text(
      "dimension 3\natom_style atomic\n",
      encoding="utf-8",
    )
    (folder / "2.txt").write_text(
      "dimension 1\natom_style full\n",
      encoding="utf-8",
    )

  def createTrainingData(self):
    temporary_directory = tempfile.TemporaryDirectory()
    folder = Path(temporary_directory.name)
    self.createDeckFiles(folder)
    data, jobs = loadDecks(self.createData(), folder)
    return temporary_directory, data, jobs

  def testLoadDeck(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "deck.txt"
      path.write_text(
        "dimension 3\n\natom_style atomic\n",
        encoding="utf-8",
      )

      deck = loadDeck(path)

      self.assertEqual(deck, "dimension 3\natom_style atomic")

  def testLoadDeckRejectsMissingFile(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "missing.txt"

      with self.assertRaisesRegex(FileNotFoundError, "Input deck not found"):
        loadDeck(path)

  def testLoadDecksLinksJobIdsAndCreatesDurations(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      self.createDeckFiles(folder)

      data, jobs = loadDecks(self.createData(), folder)

      self.assertEqual(
        data.loc[data["job_id"] == 1, "deck"].iloc[0],
        "dimension 3\natom_style atomic",
      )
      self.assertEqual(
        data.loc[data["job_id"] == 2, "deck"].iloc[0],
        "dimension 1\natom_style full",
      )
      self.assertEqual(jobs["job_id"].tolist(), [1, 2])
      self.assertEqual(jobs["duration_s"].tolist(), [60.0, 120.0])

  def testCheckDataframeRequiresTwoJobs(self) -> None:
    data = pd.DataFrame([
      {
        "job_id": 1,
        "deck": "dimension 3",
        "time_rel_s": 0.0,
      },
      {
        "job_id": 1,
        "deck": "dimension 3",
        "time_rel_s": 60.0,
      },
    ])

    with self.assertRaisesRegex(ValueError, "At least two jobs"):
      checkDataframe(data)

  def testTrainModelCreatesTwoModels(self) -> None:
    temporary_directory, data, jobs = self.createTrainingData()
    self.addCleanup(temporary_directory.cleanup)

    models = trainModel(data, jobs)

    self.assertIn("duration_model", models)
    self.assertIn("usage_model", models)
    self.assertIsNot(models["duration_model"], models["usage_model"])
    self.assertEqual(models["sample_period_s"], 60.0)

  def testPredictModelReturnsCpuAndMemoryThroughTime(self) -> None:
    temporary_directory, data, jobs = self.createTrainingData()
    self.addCleanup(temporary_directory.cleanup)
    models = trainModel(data, jobs)

    prediction = predictModel(
      models,
      "dimension 1\natom_style full",
    )

    self.assertEqual(
      prediction.columns.tolist(),
      ["time_s", "cpu_cores_used", "Active"],
    )
    self.assertGreater(len(prediction), 0)
    self.assertEqual(prediction["time_s"].iloc[0], 0.0)
    self.assertTrue(prediction["time_s"].is_monotonic_increasing)
    self.assertTrue(prediction["cpu_cores_used"].notna().all())
    self.assertTrue(prediction["Active"].notna().all())

  def testDisplayPrediction(self) -> None:
    prediction = pd.DataFrame({
      "time_s": [0.0],
      "cpu_cores_used": [2.0],
      "Active": [100.0],
    })
    output = StringIO()

    with redirect_stdout(output):
      displayPrediction(prediction)

    self.assertIn("M, prediction:", output.getvalue())
    self.assertIn("cpu_cores_used", output.getvalue())
    self.assertIn("Active", output.getvalue())


if __name__ == "__main__":
  unittest.main()
