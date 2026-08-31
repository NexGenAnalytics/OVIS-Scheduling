# modeling

## Description

- Valid training profile should contains `deck.txt`, `cpu.txt`, and `memory.txt`.
- Each CPU/memory line represents one minute.
- CPU and memory must have the same number of lines within one profile.
- Different profiles may have different durations.

## How to use

```bash
(.venv) modeling --train {path/to/training/folder}
(.venv) modeling --test {path/to/file}
```

## Examples

```bash
(.venv) modeling --train data/profiles/
(.venv) modeling --test data/input-decks/LammpsNemd/in.nemd
```
