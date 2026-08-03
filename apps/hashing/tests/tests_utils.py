import unittest
from pathlib import Path

from hashing.utils import create_hash

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

class TestUtils(unittest.TestCase):

  def test_create_hash_uses_weights(self) -> None:
    input_deck = REPOSITORY_ROOT / "data/input-decks/LammpsPeptide"
    input_path = input_deck / "in.peptide"
    weights_path = input_deck / "weights.txt"

    # unw = unweighted
    unw_normal, unw_hash = create_hash(input_path, "lammps_in_files")
    # w = weighted
    w_normal, w_hash = create_hash(
      input_path,
      "lammps_in_files",
      weights_path,
    )

    self.assertEqual(unw_normal, w_normal)
    self.assertNotEqual(unw_hash, w_hash)
