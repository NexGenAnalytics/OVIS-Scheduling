from common.models import DeckConcepts, Feature

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

def define_deck(inputdeck: list[Feature]) -> list[float]:
  """
  Return a stable, entirely numeric model vector for an input deck.
  """
  deck = DeckConcepts(
    total_steps=find_total_steps(inputdeck),
    units=find_units(inputdeck),
    atom_style=find_atom_style(inputdeck),
  )
  # print(deck)

  # NumPy expects one flat row: * is to unpack.
  return [
    *deck.total_steps,
    *_normalize_units(deck.units),
    *_normalize_style(deck.atom_style)
  ]
