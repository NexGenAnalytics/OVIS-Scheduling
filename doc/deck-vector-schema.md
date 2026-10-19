# Deck vector schema

Replace exact variable-name extraction with normalized concepts such as:

| Concept | Proposed meaning |
|---|---|
| `total_steps` | Sum of every resolved `run` command. |
| `units` | Value of the LAMMPS `units` command. |
| `atom_style` | Value of the LAMMPS `atom_style` command. |
| `pair_style` | Fixed multi-hot encoding of styles used by `pair_style` commands, including switches between phases and known hybrid substyles. Separate `other` and `missing` flags. Cutoff parameters are excluded. |
| `density` | Initial LJ lattice reduced density, followed by a presence flag. Resolves numeric literals and scalar `equal`/`index` variable references regardless of variable name. Missing or unresolved values produce `[0.0, 0.0]`. |
| `simulation_family` | Fixed one-hot encoding of a coarse workload family inferred from commands, without relying on deck filenames or comments. See classification rules below. |
| `interaction_cutoff` | Largest resolved, explicitly configured cutoff distance across supported `pair_style` and `pair_coeff` commands and phases, followed by a presence flag. Missing values produce `[0.0, 0.0]`. |
| `atom_mapping` | Fixed one-hot encoding of atom-map policy: `none`, `array`, `hash`, `auto`, `other`, `missing`. |
| `long_range_settings` | Solver category, requested accuracy, interpolation orders, explicit Coulombic/dispersion meshes, and explicit dispersion force tolerances before the first `run`. |

Makes different decks comparable.

Vector order: total steps and presence flag (2), units (2), atom style (5),
pair styles (25), density and presence flag (2), simulation family (9), interaction
cutoff and presence flag (2), atom mapping (6), long-range settings (29). Training
and prediction both append the initial particle count, giving **83 model inputs**.

## `simulation_family`

The family categories, in encoding and classification precedence order, are:

| Family | Matching deck commands |
|---|---|
| `granular` | A `gran/*` pair style or `fix pour`. |
| `rigid` | A `rigid` or `rigid/*` fix. |
| `bonded` | An explicit non-`none` `bond_style`, including the polymer decks. |
| `particle_exchange` | A `deposit` or `evaporate` fix. |
| `nonequilibrium` | A `deform`, `viscosity`, `thermal/conductivity`, `addforce`, or `nvt/sllod` fix. |
| `many_body` | A `sw` or `tersoff` pair style, including known hybrid substyles. |
| `electrostatic` | A Coulombic or `yukawa` pair style, or a non-`none` `kspace_style`. |
| `simple_fluid` | Another pair style supported by the cutoff extractor (LJ, Morse, Mie, soft). |
| `other` | No rule matches. |
