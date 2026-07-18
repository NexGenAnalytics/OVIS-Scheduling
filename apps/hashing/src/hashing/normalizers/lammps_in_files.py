def normalize_lammps(lines: list[str]) -> set[str]:
  """
  Method:
  - Normalize LAMMPS lines into {'command=value', ...}.

  Example 1:
  - Input: ["# title", "dimension 5", "radius 1 2 3]
  - Output: {"dimension=5", "radius=1 2 3"}
  """
  tokens = set()
  continuation = ""

  for line in lines:

    # clean whitespace
    line = line.strip()

    # avoid empty and comment lines
    if not line or line.startswith("#"):
      continue

    if continuation:
      line = f"{continuation} {line}"

    # save incomplete commands for the next line
    if line.rstrip().endswith("&"):
      continuation = line.rstrip()[:-1].strip()
      continue

    # reset
    continuation = ""

    # split command and value
    parts = line.split(maxsplit=1)

    # ignore lines that do not contrain both a command and a value
    if len(parts) != 2:
      continue

    # build token
    command, value = parts
    tokens.add(f"{command}={value}")

  return tokens
