# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

## Apps

Entry point for an app is `apps/[name]/src/[name]/cli.py`.

### modeling_training

- `benchmark.py`: Train models and compare results
- `dataset.py`: Loading, feature extraction, targets, splitting
- `evaluation.py`: Metrics and result data structures
- `models.py`: Dummy, Ridge, MLP model factories

### modeling_prediction

Expected CPU and memory for this input deck under the reference configuration.

Reference configuration are:
- `nodes` == 1
- `ntasks` == 48
- `ntasks_per_node` == 48
- `omp_num_threads` == 1

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
```

# Results

Understand the results:
```
+25%  model has 25% less error than Dummy
  0%  equivalent to Dummy
-15%  model has 15% more error than Dummy
```

After training:
```txt
---------------- RESOURCE: cpu ----------------
MODEL     TARGET               MAE          RMSE  MAE VS DUMMY    RMSE VS DUMMY
dummy     max               0.0095        0.0107          0.0%          0.0%
dummy     mean              0.0071        0.0078          0.0%          0.0%
dummy     min               0.0043        0.0058          0.0%          0.0%
ridge     max               0.0164        0.0171        -71.9%        -59.9%
ridge     mean              0.0046        0.0056         35.8%         28.2%
ridge     min               0.0039        0.0054          9.7%          6.4%
mlp       max               0.0197        0.0203       -106.8%        -90.0%
mlp       mean              0.0067        0.0075          5.8%          4.7%
mlp       min               0.0040        0.0056          7.0%          3.8%
Selected model: ridge
Saved ridge to output/models/cpu_model.joblib
```

After prediction:
```txt
Model: output/models/cpu_model.joblib
{'max': 4.316, 'mean': 4.035, 'min': 4.000}
```
