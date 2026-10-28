"""Fixed resource concepts from a straight-line LAMMPS deck, without executing it.

Sample settings at run/minimize boundaries. Numeric ranges retain phase changes;
style counts retain simultaneous workloads. IDs, filenames and seeds are never
model inputs. Unsupported syntax stays unknown rather than becoming a hash.
"""

from collections import Counter
import re

from common.models import Feature
from common.scalars import resolve_scalar, resolve_token, update_variable
from common.settings import _options

FIX_STYLES = (
  "nve", "nve/limit", "nve/sphere", "nvt", "npt", "nph", "nvt/sllod",
  "langevin", "temp/berendsen", "temp/rescale", "rigid", "rigid/small",
  "rigid/nve", "rigid/nvt", "rigid/npt", "shake", "rattle", "deform",
  "balance", "momentum", "addforce", "setforce", "aveforce", "gravity",
  "wall/reflect", "wall/lj93", "wall/lj126", "wall/gran", "indent",
  "deposit", "evaporate", "pour", "viscosity", "thermal/conductivity",
  "enforce2d", "adapt", "ave/time", "ave/chunk", "ave/atom", "vector",
  "store/state", "property/atom", "other",
)
COMPUTE_STYLES = (
  "temp", "temp/profile", "temp/deform", "pressure", "pe", "pe/atom",
  "ke/atom", "stress/atom", "centroid/stress/atom", "msd", "rdf", "chunk/atom",
  "reduce", "reduce/chunk", "com", "coord/atom", "voronoi/atom",
  "erotate/sphere", "other",
)

# Physical coefficients only; exclude optional cutoff distances and file paths.
PAIR_COEFFICIENT_COUNTS = {
  **dict.fromkeys(("lj/cut", "lj/cut/coul/cut", "lj/cut/coul/debye", "lj/cut/coul/dsf",
                  "lj/cut/coul/long", "lj/gromacs", "lj/long/coul/long",
                  "lj/smooth/linear", "lj96/cut", "dpd"), 2),
  "lj/expand": 3, "mie/cut": 4, "morse": 3, "soft": 1, "yukawa": 1,
  "born/coul/long": 5, "buck": 3, "buck/coul/cut": 3, "buck/coul/long": 3,
  "dpd/tstat": 1,
}

# Each numeric concept uses [minimum, maximum, present] over sampled phases.
# Ranges summarize explicit settings; omitted defaults remain unknown.
NUMERIC_GROUPS = {
  "neighbor_settings": (
    "skin", "delay", "every", "check", "once", "page", "one", "binsize",
    "exclusions", "comm_cutoff", "comm_vel", "sort_interval", "sort_binsize",
    "newton_pair", "newton_bond", "processors_x", "processors_y", "processors_z",
  ),
  "geometry_settings": (
    "dimension", "lattice_scale", "box_x", "box_y", "box_z",
    "tilt_xy", "tilt_xz", "tilt_yz", "atom_types", "bond/types", "angle/types",
    "dihedral/types", "improper/types", "extra/bond/per/atom", "extra/angle/per/atom",
    "extra/dihedral/per/atom", "extra/improper/per/atom", "extra/special/per/atom",
  ),
  "interaction_settings": (
    "mass", "coefficient_1", "coefficient_2", "coefficient_3", "coefficient_4",
    "coefficient_5", "bond_coefficient_1", "bond_coefficient_2",
    "bond_coefficient_3", "bond_coefficient_4", "charge", "diameter",
    "particle_density", "type_fraction", "topology_spacing", "pair_shift", "pair_tail", "pair_table",
    "special_lj_1", "special_lj_2", "special_lj_3",
    "special_coul_1", "special_coul_2", "special_coul_3",
  ),
  "dynamics_settings": (
    "timestep", "velocity_temperature", "temperature_start", "temperature_stop",
    "temperature_damping", "pressure_start", "pressure_stop", "pressure_damping",
    "thermostat_chain", "barostat_chain", "thermostat_loops", "barostat_loops",
    "fix_interval", "fix_samples", "fix_period", "deform_rate",
    "deposit_count", "deposit_interval", "evaporate_count", "evaporate_interval",
    "pour_count", "pour_diameter", "force_x", "force_y", "force_z",
    "min_etol", "min_ftol", "min_iterations", "min_evaluations",
  ),
  "output_settings": (
    "thermo_interval", "thermo_columns", "thermo_flush", "dump_interval", "dump_count",
    "dump_columns", "dump_buffer", "restart_interval", "rdf_bins", "chunk_width",
  ),
}

CATEGORY_GROUPS = {
  "neighbor_settings": {
    "neighbor_style": ("bin", "nsq", "multi", "other", "missing"),
    "comm_style": ("brick", "tiled", "other", "missing"),
    "comm_mode": ("single", "multi", "other", "missing"),
  },
  "geometry_settings": {
    "lattice_style": ("none", "sc", "bcc", "fcc", "hcp", "diamond", "sq", "sq2", "hex", "custom", "other", "missing"),
    "box_style": ("block", "prism", "other", "missing"),
    "box_units": ("lattice", "box", "other", "missing"),
    **{f"boundary_{axis}": ("p", "f", "s", "m", "ff", "fs", "fm", "sf", "ss", "sm", "mf", "ms", "mm", "other", "missing") for axis in "xyz"},
  },
  "interaction_settings": {
    **{command: ("none", "harmonic", "fene", "fene/expand", "morse", "class2", "charmm", "opls", "cvff", "hybrid", "other", "missing")
       for command in ("bond_style", "angle_style", "dihedral_style", "improper_style")},
    "pair_mix": ("geometric", "arithmetic", "sixthpower", "other", "missing"),
    "accelerator": ("gpu", "kk", "omp", "intel", "opt", "other", "missing"),
  },
  "dynamics_settings": {
    "run_style": ("verlet", "respa", "respa/omp", "other", "missing"),
    "min_style": ("cg", "sd", "hftn", "quickmin", "fire", "spin", "other", "missing"),
    "velocity_distribution": ("uniform", "gaussian", "other", "missing"),
    "velocity_loop": ("all", "local", "geom", "other", "missing"),
  },
  "output_settings": {
    "thermo_style": ("one", "multi", "custom", "yaml", "other", "missing"),
    "dump_style": ("atom", "custom", "xyz", "dcd", "local", "cfg", "image", "movie", "other", "missing"),
  },
}

EXECUTION_FIELDS = ("run_count", "zero_run_count", "unresolved_run_count", "minimize_count",
                    "dynamic_control", "external_input")


def workload_feature_names() -> dict[str, tuple[str, ...]]:
  names = {"execution": EXECUTION_FIELDS}
  for group, fields in NUMERIC_GROUPS.items():
    columns = [f"{field}.{stat}" for field in fields for stat in ("min", "max", "present")]
    columns.extend(f"{field}.{category}" for field, categories in CATEGORY_GROUPS[group].items()
                   for category in categories)
    if group == "dynamics_settings":
      columns.extend(f"fix.{style}.max_count" for style in FIX_STYLES)
      columns.extend(f"compute.{style}.max_count" for style in COMPUTE_STYLES)
    names[group] = tuple(columns)
  return names


def _base_style(value: str) -> str:
  for suffix in ("/gpu", "/kk", "/omp", "/intel", "/opt"):
    value = value.removesuffix(suffix)
  return value


def define_workload(inputdeck: list[Feature]) -> dict[str, list[float]]:
  variables: dict[str, str] = {}
  numbers: dict[str, list[float | None]] = {}
  categories: dict[str, set[str]] = {}
  samples: dict[str, list[float]] = {}
  category_samples: dict[str, set[str]] = {}
  active: dict[str, dict[str, list[str]]] = {command: {} for command in ("fix", "compute", "dump")}
  peaks = {"fix": Counter(), "compute": Counter()}
  coefficients: dict[str, dict[tuple[str, ...], list[str]]] = {key: {} for key in ("mass", "pair_coeff", "bond_coeff", "set")}
  regions: dict[str, list[str]] = {}
  atom_variables: dict[str, str] = {}
  execution = dict.fromkeys(EXECUTION_FIELDS, 0.0)
  pair_style = ""
  pair_instances: Counter[str] = Counter()
  sampled = False

  def put(field: str, *values: str) -> None:
    numbers[field] = [resolve_scalar(value, variables) for value in values]

  def category(field: str, *values: str) -> None:
    categories[field] = set(values)

  def options(arguments: list[str], mapping: dict[str, str]) -> None:
    for key, values in _options(arguments, dict.fromkeys(mapping, 1)):
      put(mapping[key], *values)

  def snapshot() -> None:
    nonlocal sampled
    sampled = True
    snapshot_numbers = {key: list(values) for key, values in numbers.items()}
    snapshot_numbers["dump_count"] = [float(len(active["dump"]))]
    snapshot_categories = {key: set(values) for key, values in categories.items()}

    def add(field: str, values: list[str]) -> None:
      snapshot_numbers.setdefault(field, []).extend(resolve_scalar(v, variables) for v in values)

    for command, definitions in active.items():
      counts: Counter[str] = Counter()
      known = FIX_STYLES if command == "fix" else COMPUTE_STYLES
      for args in definitions.values():
        style, values = _base_style(args[1]), args[2:]
        if style != args[1]:
          snapshot_categories.setdefault("accelerator", set()).add(args[1].split("/")[-1])
        counts[style if style in known else "other"] += 1
        if command == "fix":
          if style in ("langevin", "temp/berendsen", "temp/rescale"):
            offset = 1 if style == "temp/rescale" else 0
            for i, field in enumerate(("temperature_start", "temperature_stop", "temperature_damping")):
              if i + offset < len(values) and not (style == "temp/rescale" and i == 2):
                add(field, [values[i + offset]])
          if style in ("nvt", "npt", "nph", "nvt/sllod"):
            arities = {"temp": 3, **dict.fromkeys(("iso", "aniso", "tri", "x", "y", "z", "xy", "xz", "yz"), 3),
                       **dict.fromkeys(("tchain", "pchain", "tloop", "ploop", "couple", "drag", "mtk", "dilate", "flip", "nreset"), 1)}
            for key, payload in _options(values, arities):
              if key == "temp" or arities[key] == 3:
                prefix = "temperature" if key == "temp" else "pressure"
                for name, value in zip(("start", "stop", "damping"), payload):
                  add(f"{prefix}_{name}", [value])
              elif key in ("tchain", "pchain", "tloop", "ploop"):
                field = {"tchain": "thermostat_chain", "pchain": "barostat_chain", "tloop": "thermostat_loops", "ploop": "barostat_loops"}[key]
                add(field, payload)
          if style in ("ave/time", "ave/chunk", "ave/atom"):
            for field, value in zip(("fix_interval", "fix_samples", "fix_period"), values[:3]):
              add(field, [value])
          elif style in ("momentum", "temp/rescale", "viscosity", "thermal/conductivity", "balance", "vector", "store/state", "adapt", "deform") and values:
            add("fix_interval", values[:1])
          if style == "deform":
            for index, value in enumerate(values[:-1]):
              if value in ("erate", "trate", "vel", "delta", "scale"):
                add("deform_rate", values[index + 1:index + 2])
          if style == "deposit":
            add("deposit_count", values[:1])
            add("deposit_interval", values[2:3])
          elif style == "evaporate":
            add("evaporate_interval", values[:1])
            add("evaporate_count", values[1:2])
          elif style == "pour":
            add("pour_count", values[:1])
            for key, payload in _options(values[3:], {"vol": 2, "region": 1, "diam": 3, "dens": 2, "id": 1, "rate": 1, "mol": 1, "rigid": 1, "shake": 1}):
              if key == "diam" and payload[0] == "range":
                add("pour_diameter", payload[1:])
          if style in ("addforce", "setforce", "aveforce"):
            for axis, value in zip("xyz", values[:3]):
              add(f"force_{axis}", [value])
        elif command == "compute":
          if style == "rdf":
            add("rdf_bins", values[:1])
          elif style == "chunk/atom" and values and values[0] == "bin/1d":
            add("chunk_width", values[3:4])
        else:
          snapshot_categories.setdefault("dump_style", set()).add(style)
          add("dump_interval", values[:1])
          if style == "custom" and len(values) >= 2:
            add("dump_columns", [str(len(values[2:]))])
      if command in peaks:
        for style, count in counts.items():
          peaks[command][style] = max(peaks[command][style], count)

    for values in coefficients["mass"].values():
      add("mass", values)
    for values in coefficients["pair_coeff"].values():
      for index, value in enumerate(values):
        add(f"coefficient_{index + 1}", [value])
    for values in coefficients["bond_coeff"].values():
      for index, value in enumerate(values[:4]):
        add(f"bond_coefficient_{index + 1}", [value])
    for key, values in coefficients["set"].items():
      add({"charge": "charge", "diameter": "diameter", "density": "particle_density", "type/fraction": "type_fraction"}[key[-1]], values)
    for field, values in snapshot_numbers.items():
      samples.setdefault(field, []).extend(v for v in values if v is not None)
    for field, values in snapshot_categories.items():
      category_samples.setdefault(field, set()).update(values)

  for feature in inputdeck:
    command = feature.command
    if command == "clear":
      numbers.clear()
      categories.clear()
      regions.clear()
      for definitions in (*active.values(), *coefficients.values()):
        definitions.clear()
      pair_style = ""
      continue
    if command == "variable":
      update_variable(feature.arguments, variables)
      if len(feature.arguments) >= 3 and feature.arguments[1] == "atom":
        atom_variables[feature.arguments[0]] = " ".join(feature.arguments[2:])
      elif feature.arguments:
        atom_variables.pop(feature.arguments[0], None)
      continue
    args = [resolve_token(value, variables) or value for value in feature.arguments]
    frozen = [str(n) if (n := resolve_scalar(v, variables)) is not None else v for v in args]
    if command in ("run", "minimize"):
      if command == "run":
        execution["run_count"] += 1
        count = resolve_scalar(args[0], variables) if args else None
        execution["zero_run_count"] += float(count == 0)
        execution["unresolved_run_count"] += float(count is None)
        if "every" in args[1:]: execution["dynamic_control"] = 1.0
      else:
        execution["minimize_count"] += 1
        for field, value in zip(("min_etol", "min_ftol", "min_iterations", "min_evaluations"), args):
          put(field, value)
      snapshot()
      continue
    if command in ("if", "jump", "next", "label", "rerun"):
      execution["dynamic_control"] = 1.0
    if command in ("include", "read_data", "read_restart"):
      execution["external_input"] = 1.0
    if not args:
      continue
    if command in active and len(args) >= 3:
      # Freeze static numeric variable references at the definition point.
      active[command][args[0]] = frozen[1:]
    elif command in ("unfix", "uncompute", "undump"):
      active[command[2:]].pop(args[0], None)
    elif command == "neighbor" and len(args) >= 2:
      put("skin", args[0])
      category("neighbor_style", args[1])
    elif command == "neigh_modify":
      index = 0
      while index < len(args):
        key = args[index]
        if key in ("delay", "every", "check", "once", "page", "one", "binsize", "cluster", "include", "bin/hash") and index + 1 < len(args):
          value = args[index + 1]
          if key not in ("cluster", "include", "bin/hash"):
            put(key, {"yes": "1", "no": "0"}.get(value, value))
          index += 2
        elif key == "exclude" and index + 1 < len(args):
          kind = args[index + 1]
          if kind == "none":
            put("exclusions", "0")
          else:
            numbers["exclusions"] = [(numbers.get("exclusions", [0.0])[0] or 0.0) + 1]
          index += {"none": 2, "type": 4, "group": 4, "molecule/intra": 3, "molecule/inter": 3}.get(kind, len(args))
        else:
          break
    elif command == "comm_modify":
      for key, values in _options(args, {"mode": 1, "cutoff": 1, "vel": 1, "group": 1, "cutoff/multi": 2, "reduce/multi": 0}):
        if key == "mode": category("comm_mode", *values)
        elif key == "cutoff": put("comm_cutoff", *values)
        elif key == "vel": put("comm_vel", {"yes": "1", "no": "0"}.get(values[0], values[0]))
    elif command == "atom_modify":
      for key, values in _options(args, {"map": 1, "id": 1, "first": 1, "sort": 2}):
        if key == "sort":
          put("sort_interval", values[0])
          put("sort_binsize", values[1])
    elif command == "newton":
      for field, value in zip(("newton_pair", "newton_bond"), (args * 2)[:2]):
        put(field, {"on": "1", "off": "0"}.get(value, value))
    elif command == "processors" and len(args) >= 3:
      for axis, value in zip("xyz", args): put(f"processors_{axis}", value)
    elif command in ("dimension", "timestep"):
      put(command, args[0])
    elif command == "boundary":
      for axis, value in zip("xyz", args): category(f"boundary_{axis}", value)
    elif command == "lattice":
      category("lattice_style", args[0])
      if len(args) >= 2: put("lattice_scale", args[1])
    elif command == "region" and len(args) >= 2:
      regions[args[0]] = frozen[1:]
    elif command == "create_box" and len(args) >= 2:
      put("atom_types", args[0])
      options(args[2:], {field: field for field in NUMERIC_GROUPS["geometry_settings"] if "/" in field})
      region = regions.get(args[1], [])
      if region:
        category("box_style", region[0])
        category("box_units", region[region.index("units") + 1] if "units" in region[:-1] else "lattice")
      if len(region) >= 7 and region[0] in ("block", "prism"):
        for index, axis in enumerate("xyz"):
          put(f"box_{axis}", f"({region[2 + 2 * index]})-({region[1 + 2 * index]})")
        if region[0] == "prism" and len(region) >= 10:
          for field, value in zip(("tilt_xy", "tilt_xz", "tilt_yz"), region[7:10]): put(field, value)
    elif command == "replicate" and len(args) >= 3:
      for axis, value in zip("xyz", args):
        length = numbers.get(f"box_{axis}", [None])[0]
        put(f"box_{axis}", f"({length})*({value})")
    elif command == "mass" and len(args) >= 2:
      coefficients["mass"][(args[0],)] = frozen[1:2]
    elif command == "pair_style":
      pair_style = _base_style(args[0])
      pair_instances = Counter(_base_style(value) for value in args[1:])
      coefficients["pair_coeff"].clear()
      if args[0] != pair_style: category("accelerator", args[0].split("/")[-1])
    elif command == "pair_coeff" and len(args) >= 3:
      values = frozen[2:]
      style = pair_style
      instance = ""
      if pair_style.startswith("hybrid"):
        style, values = _base_style(values[0]), values[1:]
        if pair_instances[style] > 1 and values:
          instance, values = values[0], values[1:]
      count = PAIR_COEFFICIENT_COUNTS.get(style, 0)
      coefficients["pair_coeff"][tuple(args[:2]) + (style, instance)] = values[:count]
    elif command == "bond_coeff" and len(args) >= 2:
      coefficients["bond_coeff"][(args[0],)] = frozen[1:]
    elif command == "set" and len(args) >= 4:
      for key, values in _options(frozen[2:], {"charge": 1, "diameter": 1, "density": 1, "type": 1, "mol": 1, "type/fraction": 3}):
        if key in ("charge", "diameter", "density"):
          coefficients["set"][tuple(args[:2]) + (key,)] = values
        elif key == "type/fraction":
          coefficients["set"][tuple(args[:2]) + (key,)] = values[1:2]
    elif command == "group" and len(args) >= 3 and args[1] == "variable":
      # A scalar modulo/divisor in a used atom-selection expression describes
      # regular topology spacing (e.g. chain ends), independent of variable IDs.
      expression = atom_variables.get(args[2], "")
      for term in re.findall(r"(?:%|/)\s*(v_[A-Za-z_][A-Za-z_0-9]*|[0-9.]+)", expression):
        value = resolve_scalar(term, variables)
        if value is not None:
          numbers.setdefault("topology_spacing", []).append(value)
    elif command == "pair_modify":
      for key, values in _options(args, dict.fromkeys(("shift", "tail", "mix", "table", "compute", "table/disp", "tabinner", "tabinner/disp", "compute/tally", "neigh/trim"), 1)):
        if key == "mix": category("pair_mix", *values)
        elif key in ("shift", "tail", "table"):
          put(f"pair_{key}", {"yes": "1", "no": "0"}.get(values[0], values[0]))
    elif command == "special_bonds":
      # Each special_bonds command resets prior weights, unlike neigh_modify.
      for kind in ("lj", "coul"):
        for index in (1, 2, 3): numbers.pop(f"special_{kind}_{index}", None)
      for key, values in _options(args, {"lj": 3, "coul": 3, "lj/coul": 3, "fene": 0, "amber": 0, "charmm": 0, "dreiding": 0, "angle": 1, "dihedral": 1, "one/five": 1}):
        presets = {"fene": ([0, 1, 1], [0, 1, 1]), "amber": ([0, 0, 0.5], [0, 0, 5/6]), "charmm": ([0, 0, 0], [0, 0, 0]), "dreiding": ([0, 0, 1], [0, 0, 1])}
        for kind in ("lj", "coul"):
          weights = presets[key][kind == "coul"] if key in presets else values if key in (kind, "lj/coul") else []
          for i, weight in enumerate(weights): put(f"special_{kind}_{i + 1}", str(weight))
    elif command == "velocity" and len(args) >= 3 and args[1] == "create":
      put("velocity_temperature", args[2])
      for key, values in _options(args[4:], dict.fromkeys(("dist", "loop", "mom", "rot", "temp", "bias", "sum", "units"), 1)):
        if key in ("dist", "loop"): category("velocity_distribution" if key == "dist" else "velocity_loop", *values)
    elif command == "thermo": put("thermo_interval", args[0])
    elif command == "thermo_style":
      category("thermo_style", args[0])
      put("thermo_columns", str(len(args) - 1) if args[0] == "custom" else "unknown")
    elif command == "thermo_modify":
      for key, values in _options(args, dict.fromkeys(("flush", "lost", "lost/bond", "norm", "temp", "press", "line"), 1)):
        if key == "flush": put("thermo_flush", {"yes": "1", "no": "0"}.get(values[0], values[0]))
    elif command == "dump_modify" and len(args) >= 3:
      for key, values in _options(args[1:], dict.fromkeys(("every", "buffer", "sort", "append", "flush", "first", "pad"), 1)):
        if key == "every" and args[0] in active["dump"]:
          active["dump"][args[0]][2] = values[0]
        elif key == "buffer": put("dump_buffer", {"yes": "1", "no": "0"}.get(values[0], values[0]))
    elif command == "restart": put("restart_interval", args[0])
    elif command in ("bond_style", "angle_style", "dihedral_style", "improper_style", "run_style", "min_style", "comm_style"):
      category(command, *(_base_style(v) for v in args))
    elif command == "suffix": category("accelerator", args[0])

  if not sampled:
    snapshot()
  result = {"execution": [execution[field] for field in EXECUTION_FIELDS]}
  for group, fields in NUMERIC_GROUPS.items():
    vector = []
    for field in fields:
      values = samples.get(field, [])
      vector.extend([min(values), max(values), 1.0] if values else [0.0, 0.0, 0.0])
    for field, choices in CATEGORY_GROUPS[group].items():
      values = category_samples.get(field, {"missing"})
      normalized = {v if v in choices else "other" for v in values}
      vector.extend(float(choice in normalized) for choice in choices)
    if group == "dynamics_settings":
      vector.extend(float(peaks["fix"][style]) for style in FIX_STYLES)
      vector.extend(float(peaks["compute"][style]) for style in COMPUTE_STYLES)
    result[group] = vector
  return result
