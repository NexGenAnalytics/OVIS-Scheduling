import unittest

from hashing.normalizers.lammps_in_files import normalize_lammps

class TestNormalizersLammps(unittest.TestCase):

  def test_lammps_in_files_1(self) -> None:
    lines = ["# title", "dimension 10 4", "size 2", "radius north south"]
    normalization = normalize_lammps(lines)
    gold = {"dimension=10 4", "size=2", "radius=north south"}
    self.assertEqual(normalization, gold)

  def test_lammps_in_files_2(self) -> None:
    lines = ["# title dimension", "10 4", "mass\n2 4"]
    normalization = normalize_lammps(lines)
    gold = {"10=4", "mass=2 4"}
    self.assertEqual(normalization, gold)

  def test_lammps_in_files_3(self) -> None:
    lines = ["# title", "## section", "size\t4 5 6"]
    normalization = normalize_lammps(lines)
    gold = {"size=4 5 6"}
    self.assertEqual(normalization, gold)

  def test_lammps_in_files_4(self) -> None:
    lines = ["# title", "## section", "size 4 5 6 &", "  7 8 9", "dimension 20.5"]
    normalization = normalize_lammps(lines)
    gold = {"size=4 5 6 7 8 9", "dimension=20.5"}
    self.assertEqual(normalization, gold)
