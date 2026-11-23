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

## Zip

| Archive            | Input deck files | Jobs with `ldms_metrics.csv` | with `ldms_baseline.csv` |
| ------------------ | ---------------- | ---------------------------- | ------------------------ |
| `ldms_data_v1.zip` | 262              | 238                          | 0                        |
| `ldms_data_v2.zip` | 266              | 540                          | 291                      |
