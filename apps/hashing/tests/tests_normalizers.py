import unittest

from hashing.normalizers import get_normalizer
from hashing.normalizers.cmake_cache_variables import normalize_cmake
from hashing.normalizers.lammps_in_files import normalize_lammps

class TestNormalizers(unittest.TestCase):

  def test_get_normalizer_cmake(self) -> None:
    method = "cmake_cache_variables"
    result = get_normalizer(method)

    gold = normalize_cmake

    self.assertEqual(result, gold)

  def test_get_normalizer_lammps(self) -> None:
    method = "lammps_in_files"
    result = get_normalizer(method)

    gold = normalize_lammps

    self.assertEqual(result, gold)
