# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

## Apps

Entry point for an app is `apps/[name]/src/[name]/cli.py`.

### modeling_training

- `benchmark.py`: Train models and compare results
- `dataset.py`: Loading, feature extraction, targets, splitting
- `evaluation.py`: Metrics and result data structures
- `models.py`: Dummy, Ridge, MLP model factories

```
+25%  model has 25% less error than Dummy
  0%  equivalent to Dummy
-15%  model has 15% more error than Dummy
```

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

(.venv) modeling-prediction --model output/models/cpu_model.joblib \
  --input data/input-decks/in.test

(.venv) modeling-prediction --model output/models/memory_model.joblib \
  --input data/input-decks/in.test

(.venv) modeling-prediction --model output/models/memory_model.joblib \
  --input data/input-decks/lammps/in.binary_lj_032k_150k
```
