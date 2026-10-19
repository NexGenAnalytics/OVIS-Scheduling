# Deck vector schema

Replace exact variable-name extraction with normalized concepts such as:

| Concept | Proposed meaning |
|---|---|
| `total_steps` | Sum of every resolved `run` command. |
| `units` | Value of the LAMMPS `units` command. |
| `atom_style` | Value of the LAMMPS `atom_style` command. |
| `pair_style` | Fixed multi-hot encoding of styles used by `pair_style` commands, including switches between phases and known hybrid substyles. Separate `other` and `missing` flags. Cutoff parameters are excluded. |
| `density` | Initial LJ lattice reduced density, followed by a presence flag. Resolves numeric literals and scalar `equal`/`index` variable references regardless of variable name. Missing or unresolved values produce `[0.0, 0.0]`. |

Makes different decks comparable.
