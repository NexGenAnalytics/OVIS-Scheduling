import hashlib

def _simhash(tokens, bits=32) -> int:
  """
  Create hash from a set[str].
  """
  vector = [0] * bits

  for token in tokens:
    h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
    for i in range(bits):
      bit = (h >> i) & 1
      vector[i] += 1 if bit else -1

  fingerprint = 0
  for i, value in enumerate(vector):
    if value > 0:
      fingerprint |= 1 << i

  return fingerprint

def create_hash(path: str, method) -> tuple[set[str], int]:
  lines = _read_file(path)
  normalizer = get_normalizer(method)
  normalization = normalizer(lines)
  simhash32 = _simhash(normalization)
  return normalization, simhash32
