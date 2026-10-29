import shlex

from .Feature import Feature

def parse_inputdeck(content: str) -> list[Feature]:
  features: list[Feature] = []
  pending = ""

  for line_number, line in enumerate(content.splitlines(), start=1):
    # LAMMPS joins lines ending in '&' before tokenizing the command.
    if line.rstrip().endswith("&"):
      pending += line.rstrip()[:-1] + " "
      continue
    line, pending = pending + line, ""
    try:
      tokens = shlex.split(line, comments=True, posix=True)
    except ValueError as error:
      raise ValueError(f"Invalid input-deck {line_number}") from error

    if not tokens:
      continue

    command, *arguments = tokens

    feature = None
    match command:
      case "units" | "atom_style" | "thermo" | "timestep":
        feature = Feature(command = command, arguments = arguments)
      case "variable":
        feature = Feature(command = arguments[0], arguments = arguments[2])
      case _:
        pass

    if feature is not None:
      features.append(feature)

  if pending:
    raise ValueError("Unterminated input-deck continuation")

  return features
