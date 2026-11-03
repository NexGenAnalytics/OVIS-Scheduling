import argparse

from .filegraph import display_graph
from .histogram import display_histogram

from visualizer.metrics import max_value, filter_metric

def define_args():
  parser = argparse.ArgumentParser(description="Plot CPU and memory usage.")

  commands = parser.add_mutually_exclusive_group(required=True)
  commands.add_argument(
    "--file", metavar="PATH", help="Plot one run over time.")
  commands.add_argument(
    "--hmemory", dest="histogram_memory", action="store_true")
  commands.add_argument(
    "--hcpu", dest="histogram_cpu", action="store_true")

  args = parser.parse_args()
  return args

def main() -> None:
  print("V, start")

  args = define_args()

  match vars(args):
    case {"file": str(path_file)}:
      display_graph(path_file)
    case {"histogram_memory": True}:
      display_histogram("memory")
    case {"histogram_cpu": True}:
      display_histogram("cpu")

  print("V, end")
