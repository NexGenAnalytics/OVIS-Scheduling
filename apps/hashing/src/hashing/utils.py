import hashlib

from common.utils import read_file
from hashing.normalizers import get_normalizer

WEIGHT_MULTIPLIER = 2

def _simhash(
  tokens: set[str],
  bits: int = 32,
  weighted_parameters: set[str] | None = None,
) -> int:
  """
  Create a hash, giving selected parameters more influence on each bit.
  """
  vector = [0] * bits

  for token in tokens:
    parameter = token.split("=", maxsplit=1)[0]
    weight = 1 # default value
    if parameter in weighted_parameters:
      weight = WEIGHT_MULTIPLIER

    h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
    for i in range(bits):
      bit = (h >> i) & 1
      vector[i] += weight if bit else -weight

  fingerprint = 0
  for i, value in enumerate(vector):
    if value > 0:
      fingerprint |= 1 << i

  return fingerprint

def _read_weights(path: str) -> set[str]:
  """
  Read one parameter name per line, ignoring blanks and comments.
  """
  weights = set()

  for line in read_file(path):
    parameter = line.split("#", maxsplit=1)[0].strip()
    if parameter:
      weights.add(parameter)

  return weights

def create_hash(
  input_path: str,
  method: str,
  weights_path: str | None = None,
) -> tuple[set[str], int]:
  lines = read_file(input_path)
  normalizer = get_normalizer(method)
  normalization = normalizer(lines)

  weighted_parameters = set()
  if weights_path:
    weighted_parameters = _read_weights(weights_path)

  simhash32 = _simhash(
    tokens=normalization,
    weighted_parameters=weighted_parameters,
  )
  return normalization, simhash32
