# NGA-OVIS

`NexGen Analytics` work for Sandia `Open-source Varnish Information System`.

Predict a LAMMPS job's peak memory, CPU and duration when it is submitted, so it
can be co-scheduled without exceeding node limits. Two approaches are compared:
regressors trained on deck features (`modeling-training`) and a similarity index
that predicts from the most similar measured runs (`similarity`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## Compare approaches

```bash
similarity evaluate --simu data/
```

Cross-validates every neighbor method against the regressors on the same targets
and folds, holding out either a deck (new variant of a known family) or a whole
family. Rows are `<distance> k=<n>` (average of the n nearest runs),
`<distance> family` (nearest run's family trend) and `regressor <model>` (see
[regression models](doc/regression-models.md)). `UNDER >20%` is the share of
large underestimates, the error that causes out-of-memory kills.

## Predict a new deck

```bash
similarity build --simu data/ --distance forest
similarity nearest --input data/input-decks/lammps/in.eam_cu_256k_080k --particle-count 256000 --k 3
```

`build` saves the index to `output/similarity/index.joblib`. `nearest` prints, per
resource, the prediction, the neighbor runs it came from, and their distance
against the no-match cutoff; beyond the cutoff it prints `no good match`.

## Data

Runs come from `data/ldms/summary_global.csv`, decks from `data/input-decks/lammps/`
and profiles from `data/ldms/cpu_mpi_omp/<problem>/<job_id>/`. Only finished runs
of the reference configuration (1 node, 48 tasks, 1 OpenMP thread) are loaded.

`similarity` measures peak memory per node as `mem_anon_kb + mem_shmem_kb` above
that node's idle baseline, and compares duration per atom-step so a neighbor's
runtime scales to the new deck's size. `modeling-training` instead predicts the
max, mean and min of raw `mem_active_kb`, so compare the two with
`similarity evaluate`, not with each app's own report.

See the [deck vector schema](doc/deck-vector-schema.md) for feature coverage and
the [feature validation comparison](doc/deck-feature-validation.md) for collision
counts.

## Apps

Entry point for an app is `apps/<name>/src/<name>/cli.py`.

- `common`: deck parsing, per-application converters (LAMMPS only) and run loading
- `similarity`: `profile.py` (measured targets), `distances.py` (`changed_params`,
  `euclidean`, `learned_weights`, `forest`), `evaluate.py` (cross-validation),
  `index.py` (saved index and queries), `family.py` (per-family trends)
- `modeling-training`: trains the regressors and saves the best per resource
- `modeling-prediction`: predicts with a saved regressor

```bash
modeling-training --simu data/
modeling-prediction --model output/models/memory_model.joblib \
  --input data/input-decks/in.test --particle-count 256000
```

## Results

`similarity evaluate` on 219 runs (Nov. 20 2026), as median error / within ±20% /
underestimated by >20%. CPU is omitted: every run uses about 48 cores.

| Held out | Method | Memory | Duration |
|---|---|---|---|
| Deck | `forest k=3` | 0.9% / 92% / 5.5% | 13.6% / 58% / 21% |
| Deck | `regressor extra_trees` | 1.3% / 90% / 5.5% | 16.4% / 57% / 22% |
| Deck | `regressor mlp` | 4.1% / 82% / 8.7% | 24.7% / 46% / 27% |
| Deck | `regressor dummy` | 16.0% / 74% / 11.9% | 94.7% / 9% / 32% |
| Family | `forest k=3` | 2.0% / 88% / 5.5% | 25.6% / 44% / 27% |
| Family | `regressor extra_trees` | 2.3% / 82% / 5.0% | 23.1% / 43% / 29% |
| Family | `regressor mlp` | 6.0% / 73% / 16.0% | 51.6% / 31% / 49% |

Most runs sit near the ~1.8 GB per-node MPI floor, so memory errors look smaller
than they will on larger decks. Repeat runs of one deck differ by up to 100% in
duration, which bounds how well duration can be predicted.

## Updates

Oct. 22 2026:
- Usable simulations: 117 -> 164
- Distinct feature vectors: 58 -> 105
- Training simulations: 81 -> 114

Oct. 27 2026:
- Usable simulations: 164
- Distinct feature vectors: 105 -> 162
- Training simulations: 114

Nov. 20 2026:
- Usable simulations: 164 -> 388 (219 with the memory baseline `similarity` needs)
- Added the `similarity` app
