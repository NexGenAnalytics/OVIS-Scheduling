# Deck vector schema

`common.define_deck()` builds one fixed numeric row from parsed LAMMPS commands.
Training and prediction call the same function and append the **initial particle
count** as the final column. No vocabulary is learned from the dataset. IDs,
comments, filenames, unused scalar variables and random seeds are excluded.

The expanded schema has **664 deck columns / 665 model inputs**. The ordered
names are exported as `common.DECK_FEATURE_NAMES`; numeric field definitions and
style vocabularies live in `common.workload`. Models saved by training carry the
complete column order in `deck_feature_names_`. Prediction rejects incompatible
models with a retraining message. **Retrain existing models before prediction.**

## Column order

| Concept | Columns | Meaning |
|---|---:|---|
| `total_steps` | 2 | Sum of statically resolved run lengths and presence flag. |
| `units` | 10 | All eight standard unit systems, `other`, `missing`. Defaults to LJ when omitted. |
| `atom_style` | 19 | Fixed multi-hot style encoding, including hybrid components, `other`, `missing`. Defaults to atomic when omitted. |
| `pair_styles` | 44 | Multi-hot styles across declarations, including known hybrid substyles, many-body potentials, DPD, Buckingham, Coulomb, `other`, `missing`. Accelerator suffixes normalize to their base style. |
| `density` | 2 | Initial LJ lattice reduced density and presence flag. Non-LJ lattice scales are distances, not reduced densities. |
| `simulation_family` | 9 | Coarse family inferred from commands using the precedence below. |
| `interaction_cutoff` | 2 | Largest supported, resolved, explicitly configured cutoff across declarations and phases; presence flag. |
| `atom_mapping` | 6 | Initial map policy: `none`, `array`, `hash`, `auto`, `other`, `missing`. |
| `long_range_settings` | 29 | Initial solver category, requested accuracy, interpolation orders, explicit Coulombic/dispersion meshes, dispersion force tolerances. |
| `execution` | 6 | Counts of run commands, zero-step runs, unresolved run arguments and minimizations; flags for dynamic control and external input. |
| `neighbor_settings` | 67 | Neighbor and communication styles, skin, rebuild delay/every/check/once, page and per-atom capacities, binsize, exclusion count, communication cutoff/velocities, sorting, Newton flags, processor grid. |
| `geometry_settings` | 119 | Lattice and box styles, lattice scale, box extents/tilts, region units, dimension, per-axis boundaries, atom/topology types, extra topology capacity per atom. |
| `interaction_settings` | 132 | Masses, supported pair/bond coefficients, charges, diameters, particle densities, type fractions, regular topology spacing, shift/tail/table/mixing, special-bond weights, topology styles and requested acceleration backend. |
| `dynamics_settings` | 171 | Timestep, initial velocity temperature/distribution/loop, thermostat and pressure targets/damping/chains/loops, fix scheduling, deformation, applied forces, insertion/removal settings, minimization tolerances/budgets, run/min styles, active fix/compute counts. |
| `output_settings` | 46 | Thermo frequency/style/columns/flush, dump style/frequency/count/columns/buffering, restart frequency, RDF bins and one-dimensional chunk width. |

## Numeric resolution and phase summaries

The original scalar concepts use `[value, present]`. New numeric concepts use
`[minimum, maximum, present]` across sampled phases and active definitions.
Unknown values are zeros with a zero presence flag, so explicit zero differs
from missing. Presence means **at least one** value resolved; it does not imply
complete knowledge of every phase. Booleans are numeric 0/1 with the same flag.

The new workload groups sample settings immediately before each `run` and
`minimize`. They retain numeric extrema, a union of categorical settings, and the
maximum simultaneous count for each supported fix/compute style. Replaced IDs,
`unfix`, `uncompute`, `undump`, and `clear` update the active state. Settings
overridden before execution or added only after the last execution are excluded
from these groups. Decks without execution commands sample their final state.
The initial atom-map and K-space features retain their existing before-first-run
semantics; pair styles and maximum cutoff retain their across-declarations
semantics.

Scalar resolution accepts numeric literals, scalar `equal`/`index` aliases,
`${name}`, `v_name`, immediate arithmetic `$(...)`, unary signs, and arithmetic
`+ - * / % ^ **`. Variable updates are processed in command order. Dollar
references in variable definitions substitute at definition time; `v_name`
references in equal expressions resolve when used. Numeric parameters are frozen
at supported command definitions. Expressions are parsed through a bounded AST;
they never execute Python, LAMMPS, shell commands, or arbitrary functions.

`total_steps` accepts literal run counts and variables of any name, follows
`reset_timestep`, and handles `run N upto` when the current step is known. A
restart makes that step unknown until explicitly reset. The total contains only
resolved contributions. `unresolved_run_count` counts unresolved `N` arguments;
a known `N` with unknown current step can still prevent resolving `upto`.
Line continuations ending in `&` are joined before tokenization.

## Supported detail and limits

- Pair coefficient slots hold the physical coefficients in their style's
  documented order: LJ epsilon/sigma, Mie epsilon/sigma/repulsive/attractive
  exponents, Morse depth/width/equilibrium distance, soft/Yukawa amplitudes,
  Born coefficients, Buckingham coefficients and DPD parameters. Optional
  cutoffs are handled separately. Known hybrid substyles and repeated substyle
  instance indices are supported. These are ranges over explicit assignments,
  not a reconstructed per-type coefficient matrix or mixing calculation.
- The cutoff extractor covers the existing LJ/Mie/Morse/soft/Yukawa/Born styles
  plus Buckingham, DPD and explicit Coulombic cutoffs. It does not guess a cutoff
  from a potential filename, table, or particle radius.
- Box extents come from the region actually referenced by `create_box` for
  `block`/`prism`, with supported scalar arithmetic and `replicate` factors.
  Values retain the region's lattice/box units; they are not converted to
  physical volume. Runtime box changes and external data-file geometry are not
  reconstructed.
- Polymer variants can expose regular topology spacing through scalar divisors
  or modulo operands in an atom-variable expression used by `group ... variable`.
  This captures chain-end selection spacing without depending on the variable's
  name. It is a structural proxy, not an inferred molecular chain length.
- Fixed vocabularies cover common styles and the bundled deck families. Unknown
  styles map to `other`; absent styles map to `missing`. Supported option parsers
  stop at unknown or incomplete syntax rather than guessing argument boundaries.
- Loops, jumps, conditionals, quoted `run every` commands, includes, runtime
  compute/fix references and arbitrary LAMMPS functions are not interpreted.
  External inputs and dynamic control receive flags, but their contents or
  execution multiplicities cannot be recovered from this deck alone. Triple
  quoted multiline commands and immediate-expression format specifiers are not
  supported. Output column counts refer to declared tokens without expanding
  wildcards. Phase order and duration weighting are not retained by extrema.

The objective is to distinguish resource-relevant configurations while treating
equivalent spellings and seed-only variants consistently. More distinct vectors
do not guarantee that every model or target improves; see the measured
[validation comparison](deck-feature-validation.md).

## Simulation family

The first matching rule wins; filenames and comments do not participate.

| Family | Matching commands |
|---|---|
| `granular` | `gran/*` or `granular` pair style, or `fix pour`. |
| `rigid` | `rigid` or `rigid/*` fix. |
| `bonded` | Explicit non-`none` `bond_style`. |
| `particle_exchange` | `deposit` or `evaporate` fix. |
| `nonequilibrium` | `deform`, `viscosity`, `thermal/conductivity`, `addforce`, or `nvt/sllod` fix. |
| `many_body` | SW, Tersoff, EAM, AIREBO/REBO, ReaxFF, MEAM, SNAP, PACE, including known hybrid substyles. |
| `electrostatic` | Coulombic/Yukawa pair style or non-`none` K-space solver. |
| `simple_fluid` | Another supported explicit-cutoff pair style. |
| `other` | No matching rule. |

Command semantics follow the official LAMMPS references for
[neighbor lists](https://docs.lammps.org/neighbor.html),
[neighbor settings](https://docs.lammps.org/neigh_modify.html),
[communication](https://docs.lammps.org/comm_modify.html),
[atom settings](https://docs.lammps.org/atom_modify.html),
[pair settings](https://docs.lammps.org/pair_modify.html),
[run lengths](https://docs.lammps.org/run.html), and
[script parsing](https://docs.lammps.org/Commands_parse.html).
