# Deck vector schema

Replace exact variable-name extraction with normalized concepts such as:

| Concept | Proposed meaning |
|---|---|
| `total_steps` | Sum of every resolved `run` command |
| `units` | Value of the LAMMPS `units` command |
| `atom_style` | Value of the LAMMPS `atom_style` command |

Makes different decks comparable.
