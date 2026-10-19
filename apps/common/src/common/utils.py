from common.models import DeckConcepts, Feature
import math

# Fixed order shared by training and prediction. Never derive it from a dataset.
PAIR_STYLES = (
  "born/coul/long", "gran/hertz/history", "gran/hooke/history",
  "hybrid", "hybrid/overlay", "hybrid/scaled",
  "lj/cut", "lj/cut/coul/cut", "lj/cut/coul/debye",
  "lj/cut/coul/dsf", "lj/cut/coul/long", "lj/cut/opt",
  "lj/expand", "lj/gromacs", "lj/long/coul/long", "lj/smooth/linear",
  "lj96/cut", "mie/cut", "morse", "soft", "sw", "tersoff", "yukawa",
  "other", "missing",
)

def find_pair_styles(inputdeck: list[Feature]) -> tuple[str, ...]:
  """Collect styles across phases, including known hybrid substyles."""
  styles = []
  for feature in inputdeck:
    if feature.command != "pair_style" or not feature.arguments:
      continue
    style = feature.arguments[0]
    styles.append(style)
    if style in ("hybrid", "hybrid/overlay", "hybrid/scaled"):
      styles.extend(arg for arg in feature.arguments[1:] if arg in PAIR_STYLES)
  return tuple(styles)

def _resolve_scalar(value: str, variables: dict[str, str]) -> float | None:
  """Resolve literals and scalar variable aliases without executing deck code."""
  seen = set()
  while True:
    try:
      number = float(value)
      return number if math.isfinite(number) else None
    except ValueError:
      if value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
      elif value.startswith("v_"):
        name = value[2:]
      else:
        return None
      if name in seen or name not in variables:
        return None
      seen.add(name)
      value = variables[name]

def find_density(inputdeck: list[Feature]) -> list[float]:
  """Return [initial LJ lattice reduced density, found], or [0, 0].

  Non-LJ lattice scales are distances, not densities. This is a lattice
  density feature, not a reconstruction of density after atom/box changes.
  """
  if find_units(inputdeck) != "lj":
    return [0.0, 0.0]

  variables: dict[str, str] = {}
  for feature in inputdeck:
    arguments = feature.arguments
    if feature.command == "variable" and len(arguments) >= 3:
      name, style, *values = arguments
      if style in ("equal", "index"):
        variables[name] = " ".join(values) if style == "equal" else values[0]
      else:
        variables.pop(name, None)
    if feature.command == "lattice" and len(arguments) >= 2:
      if arguments[0] == "none":
        continue
      density = _resolve_scalar(arguments[1], variables)
      if density is not None and density > 0:
        return [density, 1.0]
      return [0.0, 0.0]
  return [0.0, 0.0]

def find_total_steps(inputdeck: list[Feature]) -> list[float]:
  """
  Sum of `nsteps`, `nstage`, `npush`, `nhot`, `n1`, etc, use in `run`command.
  Return: [total, found?]

  [160000.0, 1.0]  # value exists
  [0.0, 0.0]       # value is missing
  """
  variables = {}

  for feature in inputdeck:
    if (
      feature.command == "variable"
      and len(feature.arguments) >= 3
      and feature.arguments[0].startswith("n")
    ):
      try:
        variables[feature.arguments[0]] = float(feature.arguments[2])
      except ValueError:
        pass

  total = 0.0
  found = False

  for feature in inputdeck:
    if feature.command == "run" and feature.arguments:
      name = feature.arguments[0].removeprefix("${").removesuffix("}")

      if name in variables:
        total += variables[name]
        found = True

  return [total, float(found)]

def find_units(inputdeck: list[Feature]) -> str:
  """
  Value of the LAMMPS `units` command.
  """
  for feature in inputdeck:
    if feature.command == "units" and feature.arguments:
      return feature.arguments[0]
  raise ValueError("Input deck contains no units command")

def find_atom_style(inputdeck: list[Feature]) -> str:
  """
  Value of the LAMMPS `atom_style` command.
  """
  for feature in inputdeck:
    if feature.command == "atom_style" and feature.arguments:
      return feature.arguments[0]
  raise ValueError("Input deck contains no atom_style command")

def _normalize_units(value: str) -> list[float]:
  """
  'lj' -> [1, 0]
  """
  UNITS_STYLES = ("lj", "metal")
  normalized = value if value in UNITS_STYLES else "other"
  return [float(normalized == category) for category in UNITS_STYLES]

def _normalize_style(value: str) -> list[float]:
  """
  'atomic'  ->  [1, 0, 0, 0, 0]
  'charge'  ->  [0, 0, 1, 0, 0]
  """
  ATOM_STYLES = ("atomic", "bond", "charge", "full", "sphere")
  normalized = value if value in ATOM_STYLES else "other"
  return [float(normalized == category) for category in ATOM_STYLES]

def _normalize_pair_styles(values: tuple[str, ...]) -> list[float]:
  normalized = {value if value in PAIR_STYLES else "other" for value in values}
  if not normalized:
    normalized.add("missing")
  return [float(category in normalized) for category in PAIR_STYLES]

def define_deck(inputdeck: list[Feature]) -> list[float]:
  """
  Return a stable, entirely numeric model vector for an input deck.
  """
  deck = DeckConcepts(
    total_steps=find_total_steps(inputdeck),
    units=find_units(inputdeck),
    atom_style=find_atom_style(inputdeck),
    pair_styles=find_pair_styles(inputdeck),
    density=find_density(inputdeck),
  )
  # print(deck)

  # NumPy expects one flat row: * is to unpack.
  return [
    *deck.total_steps,
    *_normalize_units(deck.units),
    *_normalize_style(deck.atom_style),
    *_normalize_pair_styles(deck.pair_styles),
    *deck.density,
  ]
