import ast
import math
import operator
import re


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
  """Resolve finite scalar arithmetic; never execute Python or LAMMPS code.

  Runtime properties (atoms, temperature, compute/fix references, functions,
  etc.) remain unknown. Bound expression size and recursion for untrusted decks.
  """
  binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
  remaining_nodes = 4096

  def number(expression: str, seen: frozenset[str]) -> float:
    if len(expression) > 4096 or len(seen) > 32:
      raise ValueError("Expression too large")
    expression = expression.strip()
    if expression.startswith("$(") and expression.endswith(")"):
      expression = expression[2:-1]
    expression = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", r"v_\1", expression)
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
      raise ValueError("Expression too large")

    def visit(node: ast.AST) -> float:
      nonlocal remaining_nodes
      remaining_nodes -= 1
      if remaining_nodes < 0:
        raise ValueError("Expression evaluation too large")
      if isinstance(node, ast.Constant) and type(node.value) in (float, int):
        result = float(node.value)
      elif isinstance(node, ast.Name) and node.id.startswith("v_"):
        name = node.id[2:]
        if name in seen or name not in variables:
          raise ValueError("Unknown or cyclic variable")
        result = number(variables[name], seen | {name})
      elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
      elif isinstance(node, ast.BinOp) and type(node.op) in binary:
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 32:
          raise ValueError("Exponent too large")
        result = binary[type(node.op)](left, right)
      else:
        raise ValueError("Not static scalar arithmetic")
      if not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ValueError("Not a finite real number")
      return float(result)

    return visit(tree.body)

  try:
    return number(value, frozenset())
  except (ValueError, SyntaxError, ArithmeticError, TypeError, RecursionError):
    return None


def update_variable(arguments: list[str], variables: dict[str, str]) -> None:
  if len(arguments) < 2:
    return
  name, style, *values = arguments
  if values and style in ("equal", "index"):
    expression = " ".join(values) if style == "equal" else values[0]

    def substitute(match: re.Match[str]) -> str:
      token = match.group(0)
      number = resolve_scalar(token, variables)
      resolved = str(number) if number is not None else (resolve_token(token, variables) or token)
      return resolved if len(resolved) <= 4096 else token

    # ${name} substitutes now; v_name in equal expressions is evaluated later.
    variables[name] = re.sub(r"\$\{[A-Za-z_][A-Za-z_0-9]*\}", substitute, expression)
  else:
    variables.pop(name, None)
