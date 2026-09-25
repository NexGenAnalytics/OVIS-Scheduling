# modeling

## Description

- Read informations from `data/ldms/<manifest>.csv`.
  - Get a list of `job`.
- Check if exists:
  - In `data/input-decks/lammps/in.<job.problem>`.
  - In `data/ldms/<job.build>/<job.problem>/<job.id>/ldms_metrics.csv`.
- Load content.

## Usage

```bash
(.venv) modeling --manifest {1}
```
