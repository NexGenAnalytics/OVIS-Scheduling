# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

## Apps

- `common`, common tools across apps (`read_file` for example).
- `database`, store predictions (from `modeling`) and hashs (from `hashing`).
- `hashing`, [README](./apps/hashing/README.md).
- `modeling`, neural network to create predictions of CPU and memory usage.
- `scheduler`, (issue #9).

## How to use

```bash
python3 -m venv .venv
source .venv/bin/activate # for Linux and Mac
.\.venv\Scripts\activate # for Windows

(.venv) pip install -r requirements.txt

(.venv) database --list-jobs
(.venv) database --list-by-method {method_name}
(.venv) database --get-id {id}
(.venv) database --get-hash-distance-between {id_A} {id_B}

(.venv) hashing --input {path/to/file} --normalizer {method_name} [--save]

# `modeling`: see in README

(.venv) scheduler --find-nearest-to {path/to/file} {method_name}

(.venv) deactivate
```

# Examples

```bash
(.venv) database --list-by-method lammps_in_files

(.venv) hashing --input data/input-decks/TrilinosDebug/config.txt --normalizer cmake_cache_variables
(.venv) hashing --input data/input-decks/LammpsObstacle/in.obstacle --normalizer lammps_in_files --save
(.venv) hashing --input data/input-decks/LammpsTracker/in.tracker --normalizer lammps_in_files

# `modeling`: see in README

(.venv) scheduler --find-nearest-to data/input-decks/LammpsNemd/in.nemd lammps_in_files
```

# Tests

- `(.venv) python [command] -v` makes the tests verbose.

```bash
(.venv) python -m unittest discover apps/common/tests
(.venv) python -m unittest discover apps/database/tests
(.venv) python -m unittest discover apps/hashing/tests
(.venv) python -m unittest discover apps/modeling/tests
```

## Infos

- Entry point for an app is `apps/[name]/src/[name]/cli.py`.

## Devs notes

- In `apps/modeling`, there are some WIP scripts.
- In `apps/scheduler`, there are some WIP and TODO scripts, to update with new archi.
