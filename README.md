# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

## How to use

```bash
python3 -m venv .venv
source .venv/bin/activate # for Linux and Mac
.\.venv\Scripts\activate # for Windows

(.venv) pip install -r requirements.txt

(.venv) hashing
(.venv) modeling --input data/input/KokkosCuda/config.txt --output data/intermediate/modeling.json
(.venv) scheduler

(.venv) deactivate
```

## Infos

- Entry point for an app is `apps/[name]/src/[name]/cli.py`.

- Each folders in `data/input/` have at least 2 files:
  + `config.txt` (`cmake` config command ), and
  + `exec.txt` (effective running command).

## Devs notes

- In `apps/hashing`, there are some WIP scripts, to finish.
- In `apps/scheduler`, there are some TODO scripts, to update with new archi.
