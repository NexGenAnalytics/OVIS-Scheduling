def normalize_lammps(lines: list[str]) -> set[str]:
  """
  Method:
  - Normalize LAMMPS lines into {'command=value', ...}.

  Example 1:
  - Input: ["# title", "dimension 5", "radius 1 2 3]
  - Output: {"dimension=5", "radius=1 2 3"}
  """
  tokens = set()

  # combines command attributes from different lines
  continuation = ""

  for line in lines:
    # remove inline comments, then clean whitespace
    line = line.split("#", maxsplit=1)[0].strip()

    # avoid empty
    if not line:
      continue

    if continuation:
      line = f"{continuation} {line}"

    # save incomplete commands for the next line
    if line.rstrip().endswith("&"):
      continuation = line.rstrip()[:-1].strip()
      continue

    # reset
    continuation = ""

    # split line
    parts = line.split(maxsplit=1)

    # standalone command (such as "clear")
    command = parts[0]
    token = f"{command}"

    # if there is a value
    if len(parts) == 2:
      value = parts[1]
      token = f"{command}={value}"

    # build token
    tokens.add(token)

  return tokens
