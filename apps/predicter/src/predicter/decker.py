
from .Feature import Feature

def define_deck(inputdeck: list[Feature]) -> list[float]:
  results = []
  for feature in inputdeck:
    match feature.command:
      case "units" | "atom_style":
        # TODO!!!
        print(f"TODO: {feature}")
      case "thermo" | "timestep":
        value = float(feature.arguments[0])
        results.append(value)
      case _:
        value = float(feature.arguments)
        results.append(value)
  return results
