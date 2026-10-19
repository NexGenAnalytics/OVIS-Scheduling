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

# Ordered from most specific to most general; first matching family wins.
SIMULATION_FAMILIES = (
  "granular", "rigid", "bonded", "particle_exchange", "nonequilibrium",
  "many_body", "electrostatic", "simple_fluid", "other",
)

# Positions in arguments AFTER the style name (or AFTER I, J for pair_coeff).
# Only cutoff distances belong here, never damping, stiffness or energy values.
_CUTOFF_POSITIONS = {
  "born/coul/long": ((0, 1), (5,)),
  "lj/cut": ((0,), (2,)),
  "lj/cut/coul/cut": ((0, 1), (2, 3)),
  "lj/cut/coul/debye": ((1, 2), (2, 3)),
  "lj/cut/coul/dsf": ((1, 2), (2, 3)),
  "lj/cut/coul/long": ((0, 1), (2,)),
  "lj/expand": ((0,), (3,)),
  "lj/gromacs": ((1,), (3,)),
  "lj/long/coul/long": ((2, 3), (2, 3)),
  "lj/smooth/linear": ((0,), (2,)),
  "lj96/cut": ((0,), (2,)),
  "mie/cut": ((0,), (4,)),
  "morse": ((0,), (3,)),
  "soft": ((0,), (1,)),
  "yukawa": ((1,), (1,)),
}

def _base_pair_style(style: str) -> str:
  for suffix in ("/gpu", "/kk", "/omp", "/intel", "/opt"):
    if style.endswith(suffix):
      return style.removesuffix(suffix)
  return style

def find_simulation_family(inputdeck: list[Feature]) -> str:
  """Infer one coarse workload family from commands, without file names."""
  styles = {
    style
    for feature in inputdeck if feature.command == "pair_style"
    for style, _ in _pair_style_sections(feature.arguments)
  }
  fixes = {
    feature.arguments[2]
    for feature in inputdeck
    if feature.command == "fix" and len(feature.arguments) >= 3
  }
  if any(style.startswith("gran/") for style in styles) or "pour" in fixes:
    return "granular"
  if any(style == "rigid" or style.startswith("rigid/") for style in fixes):
    return "rigid"
  if any(feature.command == "bond_style" and feature.arguments
         and feature.arguments[0] != "none" for feature in inputdeck):
    return "bonded"
  if fixes & {"deposit", "evaporate"}:
    return "particle_exchange"
  if fixes & {"deform", "viscosity", "thermal/conductivity", "addforce", "nvt/sllod"}:
    return "nonequilibrium"
  if styles & {"sw", "tersoff"}:
    return "many_body"
  if (any("coul" in style or style == "yukawa" for style in styles)
      or any(feature.command == "kspace_style" and feature.arguments
             and feature.arguments[0] != "none" for feature in inputdeck)):
    return "electrostatic"
  if styles & _CUTOFF_POSITIONS.keys():
    return "simple_fluid"
  return "other"

def _pair_style_sections(arguments: list[str]) -> list[tuple[str, list[str]]]:
  """Separate known hybrid substyles, keeping their arguments together."""
  if not arguments:
    return []
  style = _base_pair_style(arguments[0])
  if style not in ("hybrid", "hybrid/overlay", "hybrid/scaled"):
    return [(style, arguments[1:])]
  starts = [i for i in range(1, len(arguments))
            if _base_pair_style(arguments[i]) in PAIR_STYLES
            and arguments[i] not in ("other", "missing")]
  sections = []
  for index, start in enumerate(starts):
    end = starts[index + 1] if index + 1 < len(starts) else len(arguments)
    if style == "hybrid/scaled" and index + 1 < len(starts):
      end -= 1  # The next substyle's scale factor is not a cutoff.
    sections.append((_base_pair_style(arguments[start]), arguments[start + 1:end]))
  return sections

def find_interaction_cutoff(inputdeck: list[Feature]) -> list[float]:
  """Return [largest resolved explicit cutoff, found] across all phases.

  Includes supported global and per-type cutoffs in the deck's distance units.
  Potential-file and particle-radius cutoffs are left missing, not guessed.
  """
  variables: dict[str, str] = {}
  cutoffs = []
  active_styles: list[str] = []
  hybrid = False

  for feature in inputdeck:
    arguments = feature.arguments
    if feature.command == "variable" and len(arguments) >= 2:
      name, style, *values = arguments
      if values and style in ("equal", "index"):
        variables[name] = " ".join(values) if style == "equal" else values[0]
      else:
        variables.pop(name, None)
      continue

    candidates = []
    if feature.command == "pair_style":
      sections = _pair_style_sections(arguments)
      active_styles = [style for style, _ in sections]
      hybrid = bool(arguments and _base_pair_style(arguments[0])
                    in ("hybrid", "hybrid/overlay", "hybrid/scaled"))
      for style, values in sections:
        positions = _CUTOFF_POSITIONS.get(style, ((), ()))[0]
        candidates.extend(values[i] for i in positions if i < len(values))
    elif feature.command == "pair_coeff" and len(arguments) >= 3:
      values = arguments[2:]
      if hybrid:
        style = _base_pair_style(values[0])
        values = values[1:]
        if active_styles.count(style) > 1:
          values = values[1:]  # Repeated substyles require an instance index.
      elif active_styles:
        style = active_styles[0]
      else:
        continue
      if style not in active_styles:
        continue
      positions = _CUTOFF_POSITIONS.get(style, ((), ()))[1]
      candidates.extend(values[i] for i in positions if i < len(values))

    for candidate in candidates:
      value = _resolve_scalar(candidate, variables)
      if value is not None and value > 0:
        cutoffs.append(value)

  return [max(cutoffs), 1.0] if cutoffs else [0.0, 0.0]

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
    simulation_family=find_simulation_family(inputdeck),
    interaction_cutoff=find_interaction_cutoff(inputdeck),
  )
  # print(deck)

  # NumPy expects one flat row: * is to unpack.
  return [
    *deck.total_steps,
    *_normalize_units(deck.units),
    *_normalize_style(deck.atom_style),
    *_normalize_pair_styles(deck.pair_styles),
    *deck.density,
    *[float(deck.simulation_family == family) for family in SIMULATION_FAMILIES],
    *deck.interaction_cutoff,
  ]
