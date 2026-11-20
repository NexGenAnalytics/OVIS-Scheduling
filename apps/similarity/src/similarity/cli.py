import argparse
from pathlib import Path

from common import get_converter, load_simulations
from .distances import DISTANCES
from .evaluate import SCENARIOS, evaluate, learned_weights, repeat_spread
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
  nearest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
  nearest_parser.add_argument("--k", type=int, default=3, help="Number of neighbors to average.")
  return parser.parse_args()

def _load_dataset(args: argparse.Namespace):
  converter = get_converter(args.converter)
  return build_dataset(load_simulations(args.simu, converter), converter), converter

def display_evaluation(args: argparse.Namespace) -> None:
  dataset, converter = _load_dataset(args)

  titles = {"deck": "new variant of a known family", "family": "unseen family"}
  results = evaluate(dataset)
  for scenario in SCENARIOS:
    print(f"\n================ HELD OUT: {titles[scenario]} ================")
    for resource, scores in results[scenario].items():
      print(f"\n{resource}")
      print(f"{'METHOD':<26}{'MEDIAN |ERR|':>14}{'WITHIN 20%':>12}{'UNDER >20%':>12}{'DIST~ERR ρ':>12}"
            f"{'NO MATCH':>10}{'MATCHED ERR':>13}")
      for score in sorted(scores, key=lambda score: score.median_error):
        correlation = "" if score.distance_error_correlation is None else f"{score.distance_error_correlation:+.2f}"
        flagged = "" if score.flagged is None else f"{score.flagged:.0%}"
        matched = "" if score.matched_median_error is None else f"{score.matched_median_error:.1%}"
        print(f"{score.method:<26}{score.median_error:>14.1%}{score.within_tolerance:>12.1%}"
              f"{score.underestimated:>12.1%}{correlation:>12}{flagged:>10}{matched:>13}")

  print("\n================ REPEAT RUNS (noise floor) ================")
  for name, difference in repeat_spread(dataset):
    spread = ", ".join(f"{resource} {value:.1%}" for resource, value in zip(RESOURCES, difference))
    print(f"{name}: {spread}")

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
      route = f"{', '.join(match.neighbors)} (distance {match.distance:.3g}, cutoff {match.cutoff:.3g})"
      if match.matched:
        print(f"{resource:<9} predicted {match.predicted:,.1f} {UNITS[resource]}  <- {route}")
      else:
        print(f"{resource:<9} no good match  <- {route}")
