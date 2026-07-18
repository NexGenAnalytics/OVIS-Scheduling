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
(.venv) database --list-by-method {method_name}
(.venv) database --get-id {id}
(.venv) database --get-hash-distance-between {id_A} {id_B}

(.venv) hashing --input {path/to/file} --normalizer {method_name}
# see examples below
# see tests below

(.venv) modeling
(.venv) scheduler

(.venv) deactivate
```

# Examples

```bash
(.venv) database --list-by-method lammps_in_files
```

## Infos

- Entry point for an app is `apps/[name]/src/[name]/cli.py`.

- Each folders in `data/input/` have at least 2 files:
  + `config.txt` (`cmake` config command ), and
  + `exec.txt` (effective running command).

## Devs notes

- In `apps/scheduler`, there are some TODO scripts, to update with new archi.
