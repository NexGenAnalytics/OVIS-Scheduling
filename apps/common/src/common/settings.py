"""Encode resource settings in effect before the first run command."""

import math

from common.models import Feature
from common.scalars import resolve_scalar, resolve_token, update_variable

ATOM_MAP_STYLES = ("none", "array", "hash", "auto", "other", "missing")
KSPACE_STYLES = ("none", "ewald", "pppm", "pppm/cg", "pppm/disp", "msm", "other")

KSPACE_FIELDS = (
  "accuracy_digits", "order", "mesh_x", "mesh_y", "mesh_z",
  "order_disp", "mesh_disp_x", "mesh_disp_y", "mesh_disp_z",
  "force_disp_real_digits", "force_disp_kspace_digits",
)


def _one_hot(value: str, categories: tuple[str, ...]) -> list[float]:
  category = value if value in categories else "other"
  return [float(category == candidate) for candidate in categories]


def _numeric_vector(fields: tuple[str, ...], settings: dict[str, float | None]) -> list[float]:
  vector = []
  for field in fields:
    value = settings[field]
    vector.extend([value, 1.0] if value is not None else [0.0, 0.0])
  return vector


def _number(value: str, *, integer: bool = False, minimum: float = 0.0) -> float | None:
  number = resolve_scalar(value, {})
  if number is None or number < minimum or (integer and not number.is_integer()):
    return None
  return number


def _accuracy(value: str) -> float | None:
  number = _number(value)
  return -math.log10(number) if number is not None and number > 0 else None


def _options(arguments: list[str], arities: dict[str, int]):
  """Consume option payloads so group IDs cannot be mistaken for keywords."""
  index = 0
  while index < len(arguments):
    key = arguments[index]
    count = arities.get(key)
    if count is None or index + count >= len(arguments):
      break  # Unknown or incomplete syntax: do not guess payload boundaries.
    yield key, arguments[index + 1:index + 1 + count]
    index += count + 1


def define_resource_settings(inputdeck: list[Feature]) -> tuple[list[float], list[float]]:
  """Return atom-map and K-space vectors using fixed column orders.

  Numeric entries are [value, known]. Defaults are known; unresolved values
  and runtime-selected meshes are missing. Later run phases are not executed.
  """
  variables: dict[str, str] = {}
  atom_style: list[str] = ["atomic"]  # LAMMPS default.
  atom_map: str | None = None
  kspace_style = "none"
  kspace: dict[str, float | None] = dict.fromkeys(KSPACE_FIELDS)
  kspace_arities = dict.fromkeys(
    ("order", "order/disp", "force/disp/real", "force/disp/kspace", "force",
     "collective", "collective/self/copy", "nonblocking", "compute", "cutoff/adjust",
     "diff", "disp/auto", "fftbench", "gewald", "gewald/disp", "minorder",
     "mix/disp", "overlap", "pressure/scalar", "slab", "splittol", "wire"), 1)
  kspace_arities.update({"mesh": 3, "mesh/disp": 3, "kmax/ewald": 3})

  for feature in inputdeck:
    if feature.command == "run":
      break
    if feature.command == "variable":
      update_variable(feature.arguments, variables)
      continue
    arguments = [resolve_token(value, variables) or value for value in feature.arguments]
    if feature.command == "atom_style" and arguments:
      atom_style = arguments
    elif feature.command == "atom_modify":
      for key, values in _options(arguments, {"map": 1, "id": 1, "first": 1, "sort": 2}):
        if key == "map":
          atom_map = {"yes": "auto", "array": "array", "hash": "hash"}.get(values[0], "other")
    elif feature.command == "kspace_style" and arguments:
      raw_style = arguments[0]
      kspace_style = raw_style
      for suffix in ("/gpu", "/kk", "/omp", "/intel", "/opt"):
        kspace_style = kspace_style.removesuffix(suffix)
      kspace = dict.fromkeys(KSPACE_FIELDS)
      if kspace_style in KSPACE_STYLES and kspace_style not in ("none", "other"):
        kspace["accuracy_digits"] = _accuracy(arguments[1]) if len(arguments) > 1 else None
        if kspace_style.startswith("pppm"):
          kspace["order"] = 7.0 if raw_style.endswith("/intel") else 5.0
          if kspace_style == "pppm/disp":
            kspace["order_disp"] = kspace["order"]
    elif feature.command == "kspace_modify" and kspace_style not in ("none", "other"):
      for key, values in _options(arguments, kspace_arities):
        if key in ("order", "order/disp"):
          field = "order" if key == "order" else "order_disp"
          if (key == "order" and (kspace_style.startswith("pppm") or kspace_style == "msm")
              or key == "order/disp" and kspace_style == "pppm/disp"):
            kspace[field] = _number(values[0], integer=True, minimum=1.0)
        elif key in ("mesh", "mesh/disp"):
          if (key == "mesh" and (kspace_style.startswith("pppm") or kspace_style == "msm")
              or key == "mesh/disp" and kspace_style == "pppm/disp"):
            mesh = [_number(value, integer=True, minimum=1.0) for value in values]
            if any(value is None for value in mesh):
              mesh = [None, None, None]
            prefix = "mesh" if key == "mesh" else "mesh_disp"
            for axis, value in zip(("x", "y", "z"), mesh):
              kspace[f"{prefix}_{axis}"] = value
        elif key in ("force/disp/real", "force/disp/kspace") and kspace_style == "pppm/disp":
          field = "force_disp_real_digits" if key.endswith("real") else "force_disp_kspace_digits"
          kspace[field] = _accuracy(values[0])

  if atom_map is None:
    molecular_styles = {"bond", "angle", "molecular", "full", "template"}
    if set(atom_style) & molecular_styles:
      atom_map = "auto"
    elif atom_style[0] in ("atomic", "charge", "sphere", "ellipsoid", "line", "tri", "body"):
      atom_map = "none"
    else:
      atom_map = "missing"

  return (
    _one_hot(atom_map, ATOM_MAP_STYLES),
    _one_hot(kspace_style, KSPACE_STYLES) + _numeric_vector(KSPACE_FIELDS, kspace),
  )
