# Expanded DeckConcepts validation

Measured on 27 October 2026 using the bundled data. The baseline extractor is
from commit `83d0c2e8c41814201acab124155362f1e2bfcb07` (83 model inputs); the
expanded extractor has 665 model inputs, including initial particle count.
The comparison tables below describe the feature expansion before the subsequent
MLP constant-feature filtering change documented separately below.

Both comparisons use the existing reference configuration (1 node, 48 MPI tasks,
1 OpenMP thread), the same 164 usable simulations, the existing seed 42 split
(114 training / 33 validation / 17 test), unchanged model factories and combined
validation selection rule. Models were selected on validation data and evaluated
on the held-out test set. No model hyperparameters were tuned for this feature
change. After validation, the normal training CLI regenerated the CPU and memory
models in `output/models/`; their previous versions were preserved in
`output/models/before-deck-concepts-bjptayt0/`. The training report is saved in
`output/deck-concepts-training.txt`.

## Collisions

Distinct model-input rows increased from **105 to 162** among 164 simulations.
This reduces the excess rows sharing a vector from 59 to 2. The remaining pairs
are `lj_base_256k_160k_seed` / `lj_base_256k_160k_wildcoeff` and
`lj_melt_032k_170k_seedA` / `lj_melt_032k_170k_seedB`: seed-only or equivalent
coefficient spelling differences are intentionally excluded.

Neighbor page capacity, skin/rebuild settings, sorting, communication cutoffs,
Newton flags, processor grids, thermostats, timesteps, temperature, mass,
interaction coefficients, output/diagnostic workloads, geometry, composition,
granular diameter ranges, and regular polymer segmentation now separate relevant
bundled variants. Tests also enforce variable/ID/filename/seed invariance.

## Held-out errors

Lower MAE and RMSE are better. CPU values use the dataset's `cpu_cores_used`
units; memory values use `mem_active_kb` (KB).

| Resource / target | Before MAE | After MAE | Before RMSE | After RMSE |
|---|---:|---:|---:|---:|
| CPU max | 0.7717 | 0.6217 | 1.5321 | 1.3589 |
| CPU mean | 1.3540 | 1.0317 | 1.7482 | 1.4132 |
| CPU min | 8.6703 | 6.6799 | 11.1452 | 8.9375 |
| Memory max | 17,915.18 | 16,286.73 | 26,709.60 | 26,793.32 |
| Memory mean | 11,555.73 | 9,091.88 | 18,501.50 | 11,057.09 |
| Memory min | 12,369.23 | 12,629.11 | 15,238.04 | 15,516.30 |

CPU mean MAE improves by approximately **23.8%**; memory peak MAE improves by
approximately **9.1%**. Peak-memory predictions within ±20% increase from
11/17 (64.7%) to 15/17 (88.2%), still below the existing 90% acceptance target.
Memory peak RMSE and minimum-memory errors worsen slightly. These are results
for one small split, not a guarantee of future accuracy.

## Model selection

| Resource | Before selected model | Before validation score | After selected model | After validation score |
|---|---|---:|---|---:|
| CPU | Extra Trees | 0.6598 | Extra Trees | 0.5434 |
| Memory | Ridge | 0.7222 | Histogram GB | 0.7597 |

The score averages MAE/RMSE relative to Dummy over all three targets. The memory
validation score worsens despite improved held-out peak/mean MAE. Ridge and MLP
do not improve consistently with the larger sparse schema; the existing
validation-based model selection remains important. Additional independent
simulations and repeated-split validation are needed to establish generalization.

## MLP constant-feature filtering

The MLP pipeline now fits `VarianceThreshold()` on training inputs before
`StandardScaler`. Prediction reuses the fitted column mask, including after
saving/loading the model. The external deck schema remains 665 inputs; only the
MLP's internal inputs change. Validation/test inputs never determine the mask.

On the same split, retraining gives the following validation MAE:

| Target | Before filter | After filter |
|---|---:|---:|
| CPU max | 2.5860 | 0.4267 |
| CPU mean | 289.5451 | 2.7447 |
| CPU min | 1,677.3844 | 25.9144 |
| Memory max, KB | 1,277,216.43 | 26,374.33 |
| Memory mean, KB | 2,067,977.84 | 24,111.27 |
| Memory min, KB | 1,201,007.59 | 36,535.99 |

The combined MLP validation score improves from 229.6620 to 2.4667 for CPU and
from 80.8697 to 0.8703 for memory. CPU still selects Extra Trees and memory still
selects Histogram GB, so the selected models' held-out results above are
unchanged. No other MLP hyperparameters were changed.

The latest training report is `output/constant-feature-training.txt`. Both model
artifacts were regenerated; their previous versions were preserved in
`output/models/before-constant-filter-kno2zf7g/`.

## Checks

All three application test suites pass:

```bash
.venv/bin/python -m unittest discover apps/common/tests
.venv/bin/python -m unittest discover apps/modeling-training/tests
.venv/bin/python -m unittest discover apps/modeling-prediction/tests
```

Coverage includes finite fixed-width vectors for the bundled decks, known-zero
versus missing values, scalar arithmetic and unsafe-expression rejection,
continuations, phase changes and object lifecycles, resource variants that
previously collided, the dataset's particle-count column, saved schema metadata,
and prediction rejection of old or reordered schemas.

Retrain to write compatible model artifacts:

```bash
.venv/bin/modeling-training --simu data/
```
