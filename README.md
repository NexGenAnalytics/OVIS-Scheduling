# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

## Datas

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

## Commands

### Setup and end working session

```bash
python3 -m venv .venv
source .venv/bin/activate # for Linux and Mac
.\.venv\Scripts\activate # for Windows
(.venv) pip install -r requirements.txt

# ... work here (see Work session) ...

(.venv) deactivate
```

### Work session

```bash
(.venv) database --list
# No jobs found.
# or
# [for each jobs] { Job( ... ) }

(.venv) database --getid [X]
# Job not found.
# or
# Job( ... )

(.venv) hasher # WIP

(.venv) predicter --train data/

(.venv) visualizer --file [problem]/[job_id]/ldms_metrics.csv
# Display graph.
```

## Architecture

- Apps are in orange.
- The database tables are blue.

```mermaid
flowchart LR

  A["LAMMPS results"] --> B["Visualizer (Draw a graph)"]:::apps

  C["Database (List job, get job by id)"]:::apps

  D["Job (ID, name, max CPU, max memory)"]:::table

  classDef apps fill:#FFA400
  classDef table fill:#02BAF7
```


## Goal :
```mermaid
flowchart TB

  C["Database"]:::apps

  D["Job (ID, name, input_deck_hash, max CPU, max memory, execution_time)"]:::table
  K["TABLE Predict (job_id, predict CPU, predict memory)"]:::table

  E["LAMMPS input deck + results"] --> F["Hasher"]:::apps
  F --> |"Store: Hash + Results"| C

  G["LAMMPS input deck"] --> F
  F --> |Hash| H["Nearer (Search by hash to get results)"]:::apps
  H --> |Compare hash| C
  C --> H
  H --> |"If value then Predict (max_cpu, max_memory)"| I["Scheduler"]:::apps
  H ---> |Else| J["Predicter (Neural network)"]:::apps
  J ---> |"Predict (max_cpu, max_memory)"| I

  classDef apps fill:#FFA400
  classDef table fill:#02BAF7
```
