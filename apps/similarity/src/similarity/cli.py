import argparse
from pathlib import Path

from common import get_converter, load_simulations
from .distances import DISTANCES
from .evaluate import evaluate, identical_deck_spread, learned_weights
from .index import build_index, load_index, query, save_index
from .profile import RESOURCES, build_dataset

DEFAULT_INDEX = Path("output/similarity/index.joblib")
UNITS = {"memory": "kB", "cpu": "cores", "duration": "s"}

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="similarity")
  parser.add_argument("--converter", default="lammps")
  commands = parser.add_subparsers(dest="command", required=True)

  evaluate_parser = commands.add_parser("evaluate", help="Compare deck distances by nearest-neighbor prediction error.")
  evaluate_parser.add_argument("--simu", type=Path, required=True)

  build_parser = commands.add_parser("build", help="Fit distances on all runs and save the index.")
  build_parser.add_argument("--simu", type=Path, required=True)
  build_parser.add_argument("--distance", choices=DISTANCES, default="forest")
  build_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)

  nearest_parser = commands.add_parser("nearest", help="Find the known runs most similar to a deck.")
  nearest_parser.add_argument("--input", type=Path, required=True)
  nearest_parser.add_argument("--particle-count", type=float, required=True)
  nearest_parser.add_argument("--k", type=int, default=3)
  nearest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
  return parser.parse_args()

def _load_dataset(args: argparse.Namespace):
  converter = get_converter(args.converter)
  return build_dataset(load_simulations(args.simu, converter), converter), converter

def display_evaluation(args: argparse.Namespace) -> None:
  dataset, converter = _load_dataset(args)

  for scenario, title in (("deck", "new variant of a known family"), ("family", "unseen family")):
    print(f"\n================ HELD OUT: {title} ================")
    for resource, scores in evaluate(dataset, scenario).items():
      print(f"\n{resource}")
      print(f"{'METHOD':<26}{'MEDIAN |ERR|':>14}{'WITHIN 20%':>12}{'UNDER >20%':>12}{'DIST~ERR ρ':>12}")
      for score in sorted(scores, key=lambda score: score.median_error):
        correlation = "" if score.distance_error_correlation is None else f"{score.distance_error_correlation:+.2f}"
        print(f"{score.method:<26}{score.median_error:>14.1%}{score.within_tolerance:>12.1%}"
              f"{score.underestimated:>12.1%}{correlation:>12}")

  print("\n================ IDENTICAL-DECK RUNS (noise proxy) ================")
  for first, second, difference in identical_deck_spread(dataset):
    spread = ", ".join(f"{name} {value:.1%}" for name, value in zip(RESOURCES, difference))
    print(f"{first} vs {second}: {spread}")

  print("\n================ LEARNED FEATURE WEIGHTS (all runs) ================")
  for resource, weights in learned_weights(dataset, converter.feature_names()).items():
    print(f"\n{resource}")
    for name, weight in weights:
      print(f"  {weight:10.4f}  {name}")

def main() -> None:
  args = init_parser()

  if args.command == "evaluate":
    display_evaluation(args)
  elif args.command == "build":
    dataset, _ = _load_dataset(args)
    save_index(build_index(dataset, args.converter, args.distance), args.index)
    print(f"Saved {args.distance} index of {len(dataset.names)} runs to {args.index}")
  else:
    index = load_index(args.index)
    matches = query(index, args.input.read_text(encoding="utf-8"), args.particle_count, args.k)
    for resource, match in matches.items():
      neighbors = ", ".join(f"{name} ({distance:.3g})" for name, distance in match.neighbors)
      print(f"{resource:<9} predicted {match.predicted:,.1f} {UNITS[resource]}  <- {neighbors}")
