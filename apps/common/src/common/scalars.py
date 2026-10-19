import math


def resolve_token(value: str, variables: dict[str, str]) -> str | None:
  """Resolve scalar variable aliases without evaluating input-deck code."""
  seen = set()
  while True:
    if value.startswith("${") and value.endswith("}"):
      name = value[2:-1]
    elif value.startswith("v_"):
      name = value[2:]
    else:
      return value
    if name in seen or name not in variables:
      return None
    seen.add(name)
    value = variables[name]


def resolve_scalar(value: str, variables: dict[str, str]) -> float | None:
  resolved = resolve_token(value, variables)
  try:
    number = float(resolved) if resolved is not None else None
  except ValueError:
    return None
  return number if number is not None and math.isfinite(number) else None


def update_variable(arguments: list[str], variables: dict[str, str]) -> None:
  if len(arguments) < 2:
    return
  name, style, *values = arguments
  if values and style in ("equal", "index"):
    variables[name] = " ".join(values) if style == "equal" else values[0]
  else:
    variables.pop(name, None)
