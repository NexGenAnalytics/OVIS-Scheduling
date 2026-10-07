from common.models import Feature

def find_variable(inputdeck: list[Feature], name: str) -> float:
  for feature in inputdeck:
    if (
      feature.command == "variable"
      and len(feature.arguments) >= 3
      and feature.arguments[0] == name
    ):
      return float(feature.arguments[2])
  raise ValueError(f"Missing numeric variable: {name}")

def define_deck_vector(inputdeck) -> list:
  deck_vector = [
    # find_atom_style(simulation.inputdeck) # TODO
    find_variable(inputdeck, "nx"),
    find_variable(inputdeck, "rho"),
    find_variable(inputdeck, "temp"),
    find_variable(inputdeck, "rc"),
    find_variable(inputdeck, "nsteps"),
  ]
  return deck_vector
