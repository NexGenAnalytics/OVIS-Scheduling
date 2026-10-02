#!/bin/bash
# Slurm batch script for one LAMMPS run; submit.py sets the cwd to the run dir and exports LMP and DECK.
set -euo pipefail
: "${LMP:?LMP is required}" "${DECK:?DECK is required}"

module purge
module load aue/gcc/12.3.0 aue/openmpi/4.1.6-gcc-12.3.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SRUN_CPUS_PER_TASK="${OMP_NUM_THREADS}"  # srun no longer inherits --cpus-per-task (Slurm >= 22.05)
omp_args=()
if [ "${OMP_NUM_THREADS}" -gt 1 ]; then
    omp_args=(-sf omp -pk omp "${OMP_NUM_THREADS}")
fi

{ env | grep '^SLURM_' | sort; module list 2>&1; } > run_metadata.txt

srun "${LMP}" ${omp_args[@]+"${omp_args[@]}"} -in "${DECK}" -log log.lammps
