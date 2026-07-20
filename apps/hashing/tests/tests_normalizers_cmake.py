import unittest

from hashing.normalizers.cmake_cache_variables import normalize_cmake

class TestNormalizersCmake(unittest.TestCase):

  def test_cmake_cache_variables_1(self) -> None:
    lines = [
      "cmake -S . -B build",
      "-DCMAKE_BUILD_TYPE=Debug",
      "-DTPL_ENABLE_MPI=ON",
    ]

    normalization = normalize_cmake(lines)

    gold = { "CMAKE_BUILD_TYPE=Debug", "TPL_ENABLE_MPI=ON" }

    self.assertEqual(normalization, gold)

  def test_cmake_cache_variables_2(self) -> None:
    lines = [
      "cmake -S . -B build",
      "-DCMAKE_BUILD_TYPE=Debug",
      "-DTPL_ENABLE_MPI=ON",
      "-DKokkos_ARCH_AMPERE80=ON"
    ]

    normalization = normalize_cmake(lines)

    gold = { "CMAKE_BUILD_TYPE=Debug", "TPL_ENABLE_MPI=ON", "Kokkos_ARCH_AMPERE80=ON" }

    self.assertEqual(normalization, gold)
