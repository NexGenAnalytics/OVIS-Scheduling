#!/bin/bash
set -euo pipefail

# Required environment variables:
#     BUILD
#     PROBLEM
#
# Optional environment variables:
#     BASE_DIR
#     NTASKS
#     OMP_NUM_THREADS

BUILD="${BUILD:?BUILD is required}"
PROBLEM="${PROBLEM:?PROBLEM is required}"

BASE_DIR="${BASE_DIR:-/gpfs/cwschil/Scheduling}"
NTASKS="${NTASKS:-4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BUILD_BASE="${BASE_DIR}/${BUILD}"
LAMMPS_BUILD="${BUILD_BASE}/build"
LAMMPS_INSTALL="${BUILD_BASE}/install"

INPUT_DECK="in.${PROBLEM}"
INPUT_DECK_PATH="${BASE_DIR}/decks/${INPUT_DECK}"

RUN_DIR="${BASE_DIR}/runs/${BUILD}/${PROBLEM}/${SLURM_JOB_ID}"

module purge
module load aue/cmake/3.29.4
module load aue/gcc/12.3.0
module load aue/openmpi/4.1.6-gcc-12.3.0

export OMP_NUM_THREADS

mkdir -p "${RUN_DIR}"

cp "${INPUT_DECK_PATH}" "${RUN_DIR}/"

{
    echo "BUILD=${BUILD}"
    echo "PROBLEM=${PROBLEM}"
    echo "BASE_DIR=${BASE_DIR}"
    echo "LAMMPS_INSTALL=${LAMMPS_INSTALL}"
    echo "INPUT_DECK=${INPUT_DECK}"
    echo "INPUT_DECK_PATH=${INPUT_DECK_PATH}"
    echo "RUN_DIR=${RUN_DIR}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
    echo "NTASKS=${NTASKS}"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo
    echo "Loaded modules:"
    module list 2>&1
} > "${RUN_DIR}/run_metadata.txt"

cd "${BASE_DIR}"

srun -n "${NTASKS}" "${LAMMPS_INSTALL}/bin/lmp" \
    -in "${RUN_DIR}/${INPUT_DECK}" \
    -log "${RUN_DIR}/log.lammps" \
    > "${RUN_DIR}/stdout.txt" \
    2> "${RUN_DIR}/stderr.txt"
