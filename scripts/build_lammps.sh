#!/bin/bash
set -euo pipefail

# -----------------------------
# User configuration
# -----------------------------
BUILD_NAME="cpu_mpi_omp"

LAMMPS_BASE="/gpfs/cwschil/Scheduling"
LAMMPS_SRC="${LAMMPS_BASE}/lammps"

BUILD_BASE="${LAMMPS_BASE}/${BUILD_NAME}"
LAMMPS_BUILD="${BUILD_BASE}/build"
LAMMPS_INSTALL="${BUILD_BASE}/install"

CMAKE_BUILD_TYPE="Release"

# Packages required by the decks in data/input-decks/lammps (OPENMP enables the -sf omp runs).
PACKAGES=(MOLECULE KSPACE MANYBODY RIGID REPLICA MISC GRANULAR OPT EXTRA-PAIR EXTRA-FIX OPENMP)

# -----------------------------
# Module environment
# -----------------------------
module purge
module load aue/cmake/3.29.4
module load aue/gcc/12.3.0
module load aue/openmpi/4.1.6-gcc-12.3.0

# -----------------------------
# Create build/install dirs
# -----------------------------
mkdir -p "${LAMMPS_BUILD}"
mkdir -p "${LAMMPS_INSTALL}"

# -----------------------------
# Record environment
# -----------------------------
ENV_LOG="${BUILD_BASE}/env.txt"

{
  echo "Build name: ${BUILD_NAME}"
  echo "Date: $(date)"
  echo
  echo "LAMMPS_SRC=${LAMMPS_SRC}"
  echo "LAMMPS_BUILD=${LAMMPS_BUILD}"
  echo "LAMMPS_INSTALL=${LAMMPS_INSTALL}"
  echo

  echo "LAMMPS git information:"
  if git -C "${LAMMPS_SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "  commit: $(git -C "${LAMMPS_SRC}" rev-parse HEAD)"
    echo "  describe: $(git -C "${LAMMPS_SRC}" describe --tags --always)"
    echo "  branch: $(git -C "${LAMMPS_SRC}" branch --show-current || true)"
    echo "  status:"
    git -C "${LAMMPS_SRC}" status --short
  else
    echo "  ${LAMMPS_SRC} is not a git checkout"
  fi
  echo

  echo "Loaded modules:"
  module list 2>&1
  echo

  echo "Compiler and tool versions:"
  echo "gcc: $(which gcc || true)"
  gcc --version || true
  echo
  echo "g++: $(which g++ || true)"
  g++ --version || true
  echo
  echo "mpicc: $(which mpicc || true)"
  mpicc --version || true
  echo
  echo "mpicxx: $(which mpicxx || true)"
  mpicxx --version || true
  echo
  echo "cmake: $(which cmake || true)"
  cmake --version || true
} > "${ENV_LOG}"

# -----------------------------
# Configure
# -----------------------------
cd "${LAMMPS_BUILD}"

PKG_FLAGS=()
for pkg in "${PACKAGES[@]}"; do
  PKG_FLAGS+=(-D "PKG_${pkg}=on")
done

cmake "${LAMMPS_SRC}/cmake" \
  -D CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -D CMAKE_INSTALL_PREFIX="${LAMMPS_INSTALL}" \
  -D BUILD_MPI=on \
  -D BUILD_OMP=on \
  "${PKG_FLAGS[@]}"

# -----------------------------
# Save CMake configuration
# -----------------------------
cp CMakeCache.txt "${BUILD_BASE}/CMakeCache.txt"

cmake -LAH . > "${BUILD_BASE}/cmake_config_full.txt"

# -----------------------------
# Build and install
# -----------------------------
cmake --build . -j

cmake --install .

# -----------------------------
# Smoke test
# -----------------------------
set +e

OMPI_MCA_mtl="^psm2" \
OMPI_MCA_pml="ob1" \
OMPI_MCA_btl="self,tcp" \
"${LAMMPS_INSTALL}/bin/lmp" -help > "${BUILD_BASE}/lmp_help.txt" 2>&1

SMOKE_STATUS=$?
set -e

if [[ ${SMOKE_STATUS} -ne 0 ]]; then
  echo "WARNING: LAMMPS smoke test failed on login node."
  echo "This may be due to MPI fabric initialization, not a build failure."
  echo "See: ${BUILD_BASE}/lmp_help.txt"
else
  echo "LAMMPS smoke test passed."
  for pkg in "${PACKAGES[@]}"; do
    sed -n '/^Installed packages:/,/^List of individual style/p' "${BUILD_BASE}/lmp_help.txt" \
      | grep -qE "(^| )${pkg}( |$)" || echo "WARNING: package ${pkg} is missing from the build."
  done
fi

echo "LAMMPS build complete."
echo "Install path: ${LAMMPS_INSTALL}"
echo "Environment log: ${ENV_LOG}"
echo "CMake cache: ${BUILD_BASE}/CMakeCache.txt"
echo "Full CMake config: ${BUILD_BASE}/cmake_config_full.txt"
