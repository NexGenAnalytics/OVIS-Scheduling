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
