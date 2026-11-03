from common.Feature import Feature

def numeric_features(inputdeck: list[Feature]) -> dict[str, float]:
  values = {}

  for feature in inputdeck:
    match feature.command:
      case "units" | "atom_style":
        continue  # Encode categorical features separately later.
      case _:
        arguments = feature.arguments
        raw = arguments[0] if isinstance(arguments, list) else arguments
        values[feature.command] = float(raw)

  return values

def define_deck(
  inputdeck: list[Feature],
  columns: list[str]
) -> list[float]:
  values = numeric_features(inputdeck)

  numeric = [values.get(name, 0.0) for name in columns]
  present = [float(name in values) for name in columns]

  return numeric + present
