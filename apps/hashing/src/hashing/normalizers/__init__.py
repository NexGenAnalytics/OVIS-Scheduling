from collections.abc import Callable

from .cmake_cache_variables import normalize_cmake
from .lammps_in_files import normalize_lammps

Normalizer = Callable[[list[str]], set[str]]

NORMALIZERS: dict[str, Normalizer] = {
  "cmake_cache_variables": normalize_cmake,
  "lammps_in_files": normalize_lammps,
}
