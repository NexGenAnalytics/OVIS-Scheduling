# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate # for Linux and Mac
.\.venv\Scripts\activate # for Windows
(.venv) pip install -r requirements.txt
```

## How to use

```bash
(.venv) modeling-training --simu {path} --deta {int} --feat {list}
# simu (simulations): path to the folder with input decks and run profiles
# deta (details): result integer because ML models needs fixed-size targets (min. 3)
# feat (features): optional, strings/variables that are impactfull on cpu and memory usage

(.venv) modeling-prediction --models {path} --input {path}
# models: path to load models
# input: path to the input deck you want to predict
```

## How to quit

```bash
(.venv) deactivate # to exit venv
```

# Examples

```bash
(.venv) modeling-training --simu data/ --deta 5

(.venv) modeling-training \
  --simu data/ \
  --deta 50 \
  --feat "variable nsteps"

(.venv) modeling-prediction \
  --models outputs/ \
  --input /data/input-decks/lammps/in.binary
```

## Infos

- Entry point for an app is `apps/[name]/src/[name]/cli.py`.
