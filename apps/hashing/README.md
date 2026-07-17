# hashing

## Description

- Create a normalization from `--input` and `--normalizer`.
- Create a hash from the normalization.
- Save normalization and hash into database.

## Current normalizers

- `cmake_cache_variables` method:
  + Use `config.txt` file.
  + Based on `-D`.
- `lammps_in_files` method:
  + Use `in.*` file.

## How do I add a new normalizer?

1. Create Python file in `hashing/normalizers/<method_name>.py`.
2. Expose the following interface: `def normalize_<name>(lines: list[str]) -> set[str]:`.
3. Add the method into `normalizers/__init__.py`:
  - Import: `from .<method_name> import normalize_<name>`.
  - Dictionary: `"<method_name>": normalize_<name>,`.
4. Add a test file in `apps/hashing/tests/`.
5. Update this `README.md`.
