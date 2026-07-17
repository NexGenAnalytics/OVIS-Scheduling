def normalize_lammps(lines: list[str]) -> set:
  """
  Method:
  - Normalize LAMMPS lines into {'command=value', ...}.

  Example 1:
  - Input: ["# title", "dimension 5", "radius 1 2 3]
  - Output: {"dimension=5", "radius=1 2 3"}
  """
  tokens = set()

  for line in lines:
    # clean whitespace
    line = line.strip().rstrip("\\").strip()

    # avoid empty and comment lines
    if not line or line.startswith("#"):
      continue

    # split command and value
    parts = line.split(maxsplit=1)
    command = parts[0]
    value = parts[1]

    # build token
    tokens.add(f"{command.strip()}={value.strip()}")

  return tokens
