# modeling

Neural networks to create predictions of CPU and memory usage.

## Description

- Read informations from `data/ldms/<manifest>.csv`.
  - Get a list of `job`.
- Check if exists:
  - In `data/input-decks/lammps/in.<job.problem>`.
  - In `data/ldms/<job.build>/<job.problem>/<job.id>/ldms_metrics.csv`.
- Load content.
- Train and save 2 models.

## Usage

```bash
(.venv) modeling --train {manifest}
(.venv) modeling --predict {input-decks}
```

## Examples

```bash
(.venv) modeling --train data/ldms/ldms_manifest_20260921.csv
(.venv) modeling --predict data/input-decks/LammpsNemd/in.nemd
```
