from argparse import ArgumentParser

from pandas import DataFrame, read_csv

from pathlib import Path

from modeling.dump_and_load_model import loadModel, saveModel
from modeling.load_and_check_data import findDataCsv, checkDataColums
from modeling.load_deck import loadDeck, loadDecks, checkDataframe
from modeling.prediction import predictModel, displayPrediction
from modeling.training import trainModel

def main() -> None:
  print("M, start")

  parser = ArgumentParser()
  commands = parser.add_mutually_exclusive_group(required=True)
  commands.add_argument("--train", metavar="PATH")
  commands.add_argument("--test", metavar="PATH")
  args = parser.parse_args()

  match vars(args):

    case {"train": str(path)}:
      folder_path = Path(path)
      file_path = findDataCsv(folder_path)

      # load data
      df = read_csv(file_path, sep=",")
      checkDataColums(df.columns)
      dataframe = DataFrame(df)

      # link job_id to <job_id>.txt
      dataframe, jobs = loadDecks(dataframe, folder_path)
      models = trainModel(dataframe, jobs)
      saveModel(models)

    case {"test": str(path)}:
      models = loadModel()
      deck = loadDeck(Path(path))
      prediction = predictModel(models, deck)
      displayPrediction(prediction)

  print("M, end")
