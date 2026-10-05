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
(.venv) modeling-training --simu {path}
# simu (simulations): path to the folder with input decks and run profiles

# Idea:
# feat (features): optional, strings/variables that are impactfull on cpu and
# memory usage
# # --feat {list}

(.venv) modeling-prediction --model {path} --input {path}
# model: path to load model
# input: path to the input deck you want to predict
```

## How to quit

```bash
(.venv) deactivate # to exit venv
```

# Examples

```bash
(.venv) modeling-training --simu data/

(.venv) modeling-training --simu data/
# [--feat "variable nsteps" ?????????!!!!!]

(.venv) modeling-prediction \
  --model output/models/cpu_model.joblib \
  --input data/input-decks/lammps/in.binary_lj_032k_150k
```

## Infos

- Entry point for an app is `apps/[name]/src/[name]/cli.py`.
