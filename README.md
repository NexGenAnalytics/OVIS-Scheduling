# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

## Apps

- `database`, store predictions (from `modeling`) and hashs (from `hashing`).
- `hashing`, [README](./apps/hashing/README.md).
- `modeling`, create a prediction of CPU and memory usage.
- `scheduler`, (issue #9).

## How to use

```bash
python3 -m venv .venv
source .venv/bin/activate # for Linux and Mac
.\.venv\Scripts\activate # for Windows

(.venv) pip install -r requirements.txt

(.venv) database --list-jobs
(.venv) hashing --input {path/to/file} --normalizer {method_name}
# see examples below
# see tests below
(.venv) modeling --input "arg_A" --output "arg_B"
(.venv) scheduler

(.venv) deactivate
```

# Examples

```bash
(.venv) hashing --input data/input-decks/TrilinosDebug/config.txt --normalizer cmake_cache_variables
(.venv) hashing --input data/input-decks/LammpsObstacle/in.obstacle --normalizer lammps_in_files
(.venv) hashing --input data/input-decks/LammpsTracker/in.tracker --normalizer lammps_in_files
```

# Tests

```bash
(.venv) python -m unittest discover apps/hashing/tests
```

## Infos

- Entry point for an app is `apps/[name]/src/[name]/cli.py`.

## Devs notes

- In `apps/modeling`, there are some WIP scripts.
- In `apps/scheduler`, there are some WIP and TODO scripts, to update with new archi.
