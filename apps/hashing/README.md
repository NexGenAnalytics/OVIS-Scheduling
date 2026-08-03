# hashing

## Description

- Create a normalization from `--input` and `--normalizer`.
- Create a hash from the normalization.
- Save (`--save`) normalization and hash into database.
- You can specify the weight (`--weight`) of the parameters you want to emphasize.

## Current normalizers

- `cmake_cache_variables` method:
  + Use `config.txt` file.
  + Based on `-D`.
- `lammps_in_files` method:
  + Use `in.*` file.

## How to use?

```bash
(.venv) hashing --input {path/to/file} \
                --normalizer {method_name} \
                [--save] \
                [--weight {path/to/file}]
```

## Examples

```bash
# normalizer: cmake_cache_variables
(.venv) hashing --input data/input-decks/TrilinosDebug/config.txt \
                --normalizer cmake_cache_variables

# normalizer: lammps_in_files
(.venv) hashing --input data/input-decks/LammpsTracker/in.tracker \
                --normalizer lammps_in_files

# save result
(.venv) hashing --input data/input-decks/LammpsObstacle/in.obstacle
                --normalizer lammps_in_files \
                --save

# use weight
(.venv) hashing --input data/input-decks/LammpsPeptide/in.peptide \
                --normalizer lammps_in_files \
                --weight data/input-decks/LammpsPeptide/weight.txt
```

## How do I add a new normalizer?

1. Create Python file in `hashing/normalizers/<method_name>.py`.
2. Expose the following interface: `def normalize_<name>(lines: list[str]) -> set[str]:`.
3. Add the method into `normalizers/__init__.py`:
  - Import: `from .<method_name> import normalize_<name>`.
  - Dictionary: `"<method_name>": normalize_<name>,`.
4. Add a test file in `apps/hashing/tests/`.
5. Update this `README.md`.
