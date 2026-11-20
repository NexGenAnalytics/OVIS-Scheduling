# Datas

- Create `/data/lammps_input_decks/`
- Create `/data/lammps_results/`
- Unzip `ldms_data.zip`
- In *input_decks* folder:
  - Put the `in.[problem]` files
- In *results* folder:
  - Put the `summary.csv` file and
  - Put the folders `[problem]/[job_id]/ldms_metrics.csv`

Example:

```
data/
-| lammps_input_decks/
----| in.A
----| in.B
-| lammps_results/
----| summary.csv
# job_id, build, problem
# 101, cpu_mpi_omp, A
# 102, cpu_mpi_omp, B
----| A/101/ldms_metrics.csv
----| B/102/ldms_metrics.csv
```
