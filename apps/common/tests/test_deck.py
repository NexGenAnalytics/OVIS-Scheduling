import math
from pathlib import Path
import unittest

from common import DECK_FEATURE_NAMES, define_deck, parse_inputdeck
from common.scalars import resolve_scalar
from common.utils import find_total_steps


BASE = """units lj
atom_style atomic
variable rho equal 0.8442
lattice fcc ${rho}
region box block 0 40 0 40 0 40
create_box 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes
fix 1 all nve
thermo 100
"""


def vector(content: str) -> list[float]:
  return define_deck(parse_inputdeck(content))


def columns(content: str) -> dict[str, float]:
  return dict(zip(DECK_FEATURE_NAMES, vector(content), strict=True))


class TestScalars(unittest.TestCase):
  def test_aliases_arithmetic_and_immediate_expressions(self):
    variables = {"size": "40", "twice": "2*v_size", "alias": "${twice}"}
    for expression, expected in (("${alias}", 80), ("$(v_twice/2)", 40),
                                 ("-${size}", -40), ("2^3 + 1", 9)):
      with self.subTest(expression=expression):
        self.assertEqual(resolve_scalar(expression, variables), expected)

  def test_dynamic_unsafe_cyclic_and_nonfinite_values_stay_unknown(self):
    for expression in ("temp", "atoms", "c_pressure[1]", "sqrt(4)", "__import__('os')",
                       "1/0", "1e309", "(-1)^0.5", "2^100000000", "v_cycle"):
      with self.subTest(expression=expression):
        self.assertIsNone(resolve_scalar(expression, {"cycle": "v_cycle"}))

  def test_repeated_expression_expansion_is_bounded(self):
    variables = {"level0": "1"}
    variables.update({f"level{i}": f"v_level{i-1}+v_level{i-1}" for i in range(1, 20)})
    self.assertIsNone(resolve_scalar("v_level19", variables))


class TestDeckConcepts(unittest.TestCase):
  def test_fixed_unique_finite_schema_for_missing_and_unknown_entries(self):
    self.assertEqual(len(DECK_FEATURE_NAMES), len(set(DECK_FEATURE_NAMES)))
    for content in ("", BASE, "units real\natom_style hybrid full sphere\nneighbor 1 mystery\nrun 0"):
      with self.subTest(content=content):
        values = vector(content)
        self.assertEqual(len(values), len(DECK_FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(v) for v in values))
    self.assertEqual(columns("units real")["units.real"], 1)
    self.assertEqual(columns("neighbor 1 mystery")["neighbor_settings.neighbor_style.other"], 1)
    self.assertEqual(columns("")["neighbor_settings.neighbor_style.missing"], 1)

  def test_total_steps_uses_command_order_and_any_variable_name(self):
    deck = """variable duration equal 2^3
run ${duration}
variable duration equal 20
run 12
run v_duration
run 0
"""
    self.assertEqual(find_total_steps(parse_inputdeck(deck)), [40.0, 1.0])

  def test_upto_and_timestep_reset_do_not_overcount(self):
    deck = "run 10\nrun 30 upto\nreset_timestep 5\nrun 15 upto\nrun 2"
    self.assertEqual(find_total_steps(parse_inputdeck(deck)), [42.0, 1.0])

  def test_unresolved_runs_and_restart_do_not_invent_step_counts(self):
    content = "read_restart restart.bin\nrun 200 upto\nrun v_unknown\nrun 10"
    self.assertEqual(find_total_steps(parse_inputdeck(content)), [10.0, 1.0])
    values = columns(content)
    self.assertEqual(values["execution.external_input"], 1)
    self.assertEqual(values["execution.unresolved_run_count"], 1)

  def test_dollar_substitution_is_frozen_but_equal_references_are_lazy(self):
    content = """variable size equal 10
variable frozen equal ${size}
variable lazy equal v_size
variable size equal 20
run v_frozen
run v_lazy
"""
    self.assertEqual(find_total_steps(parse_inputdeck(content)), [30.0, 1.0])

  def test_continued_commands_match_single_line(self):
    continued = BASE + "neigh_modify delay 0 &\n every 10 check no &\n page 200000\nrun 100"
    single = BASE + "neigh_modify delay 0 every 10 check no page 200000\nrun 100"
    self.assertEqual(vector(continued), vector(single))
    with self.assertRaises(ValueError): parse_inputdeck("fix 1 all &")

  def test_neighbor_memory_and_rebuild_settings_are_separate_concepts(self):
    values = columns(BASE + "neigh_modify delay 5 every 10 check no one 200000 page 20000000\nrun 100")
    for field, expected in (("delay", 5), ("every", 10), ("check", 0), ("one", 200000), ("page", 20000000)):
      self.assertEqual(values[f"neighbor_settings.{field}.max"], expected)
      self.assertEqual(values[f"neighbor_settings.{field}.present"], 1)

  def test_option_payloads_are_not_parsed_as_keywords(self):
    values = columns(BASE + "neigh_modify exclude group every delay check no\ncomm_modify group cutoff vel yes\nrun 100")
    self.assertEqual(values["neighbor_settings.every.max"], 1)
    self.assertEqual(values["neighbor_settings.delay.max"], 0)
    self.assertEqual(values["neighbor_settings.check.max"], 0)
    self.assertEqual(values["neighbor_settings.comm_cutoff.present"], 0)
    self.assertEqual(values["neighbor_settings.comm_vel.max"], 1)

  def test_missing_numeric_value_differs_from_known_zero(self):
    self.assertNotEqual(vector(BASE + "run 100"), vector(BASE + "atom_modify sort 0 0\nrun 100"))
    values = columns(BASE + "atom_modify sort 0 0\nrun 100")
    self.assertEqual(values["neighbor_settings.sort_interval.max"], 0)
    self.assertEqual(values["neighbor_settings.sort_interval.present"], 1)

  def test_numeric_ranges_cover_later_run_phases(self):
    values = columns(BASE + "timestep 0.001\nrun 50\ntimestep 0.01\nneighbor 1 nsq\nrun 50")
    self.assertEqual(values["dynamics_settings.timestep.min"], 0.001)
    self.assertEqual(values["dynamics_settings.timestep.max"], 0.01)
    self.assertEqual(values["neighbor_settings.neighbor_style.bin"], 1)
    self.assertEqual(values["neighbor_settings.neighbor_style.nsq"], 1)

  def test_overridden_and_unused_post_run_settings_are_ignored(self):
    self.assertEqual(vector(BASE + "timestep 1\ntimestep 0.01\nrun 50\ntimestep 2"),
                     vector(BASE + "timestep 0.01\nrun 50"))

  def test_fix_compute_lifecycles_and_simultaneous_counts(self):
    values = columns(BASE + """fix extra all langevin 1 2 0.5 12345
compute first all stress/atom NULL
compute second all stress/atom NULL
uncompute first
run 100
unfix extra
run 100
""")
    self.assertEqual(values["dynamics_settings.fix.langevin.max_count"], 1)
    self.assertEqual(values["dynamics_settings.compute.stress/atom.max_count"], 1)
    self.assertEqual(values["dynamics_settings.temperature_start.min"], 1)
    self.assertEqual(values["dynamics_settings.temperature_stop.max"], 2)

  def test_unfixed_work_is_not_counted(self):
    self.assertEqual(vector(BASE + "fix extra all langevin 1 2 0.5 12345\nunfix extra\nrun 100"),
                     vector(BASE + "run 100"))

  def test_fix_parameters_are_frozen_when_defined(self):
    values = columns(BASE + "variable heat equal 1\nfix extra all langevin ${heat} ${heat} 0.5 12345\nvariable heat equal 4\nrun 100")
    self.assertEqual(values["dynamics_settings.temperature_start.max"], 1)

  def test_thermostat_pressure_and_chain_parameters(self):
    values = columns(BASE + "unfix 1\nfix 1 all npt temp 1 2 0.5 iso -1 3 5 tchain 7 tloop 2\nrun 100")
    for field, expected in (("temperature_start", 1), ("temperature_stop", 2), ("temperature_damping", 0.5),
                            ("pressure_start", -1), ("pressure_stop", 3), ("pressure_damping", 5), ("thermostat_chain", 7), ("thermostat_loops", 2)):
      self.assertEqual(values[f"dynamics_settings.{field}.max"], expected)

  def test_renaming_variables_ids_files_and_seeds_does_not_add_identity(self):
    a = BASE + "velocity all create 1 12345 loop geom\ndump d all custom 100 a.dump id x y z\nrun 100"
    b = a.replace("rho", "reduced_density").replace("fix 1 all", "fix renamed all").replace("12345", "54321").replace("dump d ", "dump other ").replace("a.dump", "different.dump")
    self.assertEqual(vector(a), vector(b))

  def test_output_diagnostics_and_dump_modify(self):
    values = columns(BASE + """compute r all rdf 200
fix avg all ave/time 10 5 100 c_r[*] file rdf.txt
dump d all custom 100 out.dump id type x y z
dump_modify d every 50 buffer no
thermo_style custom step temp c_r[1]
thermo_modify flush yes
restart 1000 out.restart
run 100
""")
    for field, expected in (("rdf_bins", 200), ("dump_interval", 50), ("dump_columns", 5), ("thermo_columns", 3), ("thermo_flush", 1), ("restart_interval", 1000)):
      self.assertEqual(values[f"output_settings.{field}.max"], expected)
    self.assertEqual(values["dynamics_settings.fix_samples.max"], 5)

  def test_geometry_uses_create_box_region_and_resolved_extents(self):
    content = """variable size equal 40
region irrelevant block 0 100 0 100 0 100
region actual block 0 $(2*v_size) 0 ${size} 0 $(v_size/2) units box
create_box 2 actual extra/bond/per/atom 4
replicate 1 2 1
run 100
"""
    values = columns(content)
    for field, expected in (("box_x", 80), ("box_y", 80), ("box_z", 20), ("atom_types", 2), ("extra/bond/per/atom", 4)):
      self.assertEqual(values[f"geometry_settings.{field}.max"], expected)
    self.assertEqual(values["geometry_settings.box_units.box"], 1)

  def test_pair_coefficients_mass_and_mie_exponents(self):
    values = columns(BASE + "mass 1 2\npair_style mie/cut 2.5\npair_coeff * * 0.5 1 14 7 2.5\nrun 100")
    self.assertEqual(values["interaction_settings.mass.max"], 2)
    self.assertEqual(values["interaction_settings.coefficient_1.max"], 0.5)
    self.assertEqual(values["interaction_settings.coefficient_3.max"], 14)
    self.assertEqual(values["interaction_settings.coefficient_4.max"], 7)

  def test_dpd_thermostat_coefficient_is_not_its_cutoff(self):
    values = columns(BASE + "pair_style dpd/tstat 1 2 3.5 12345\npair_coeff * * 0.5 4\nrun 100")
    self.assertEqual(values["interaction_settings.coefficient_1.max"], 0.5)
    self.assertEqual(values["interaction_settings.coefficient_2.present"], 0)
    self.assertEqual(values["interaction_cutoff"], 4)

  def test_special_bonds_replacement_does_not_accumulate_weights(self):
    values = columns(BASE + "special_bonds lj 0 1 1\nspecial_bonds coul 0 0 1\nrun 100")
    self.assertEqual(values["interaction_settings.special_lj_2.present"], 0)
    self.assertEqual(values["interaction_settings.special_coul_3.max"], 1)

  def test_granular_exchange_and_composition_parameters(self):
    values = columns(BASE + """fix ins all pour 120000 1 12345 vol 0.2 50 region slab diam range 0.7 1.2
fix dep all deposit 1000 1 20 54321 region source
fix evap all evaporate 100 10 slab 99999
set type 1 type/fraction 2 0.5 10101
run 100
""")
    for field, expected in (("pour_count", 120000), ("pour_diameter", 1.2), ("deposit_count", 1000), ("deposit_interval", 20), ("evaporate_count", 10), ("evaporate_interval", 100)):
      self.assertEqual(values[f"dynamics_settings.{field}.max"], expected)
    self.assertEqual(values["interaction_settings.type_fraction.max"], 0.5)

  def test_minimization_and_unknown_runtime_values(self):
    values = columns(BASE + "timestep v_runtime\nmin_style fire\nminimize 0 1e-8 5000 50000")
    self.assertEqual(values["execution.minimize_count"], 1)
    self.assertEqual(values["dynamics_settings.min_iterations.max"], 5000)
    self.assertEqual(values["dynamics_settings.min_etol.present"], 1)
    self.assertEqual(values["dynamics_settings.timestep.present"], 0)

  def test_accelerated_styles_keep_physics_and_execution_backend(self):
    values = columns(BASE + "pair_style lj/cut/omp 3.5\nrun 100")
    self.assertEqual(values["pair_style.lj/cut"], 1)
    self.assertEqual(values["interaction_cutoff"], 3.5)
    self.assertEqual(values["interaction_settings.accelerator.omp"], 1)

  def test_bundled_resource_variants_no_longer_collide(self):
    directory = Path(__file__).resolve().parents[3] / "data/input-decks/lammps"
    baseline = vector((directory / "in.lj_base_256k_160k").read_text())
    variants = ("skin01", "skin10", "every10", "delay5", "neighpage", "nosort", "commcut", "newtonoff", "procs2d",
                "dt0025", "temp01", "nvt", "npt", "nph", "langevin", "berendsen", "rescale", "thermo1", "dump",
                "msd", "rdf", "stressatom", "mass2", "eps05", "shift", "tail", "slabshape", "bcc")
    for variant in variants:
      with self.subTest(variant=variant):
        self.assertNotEqual(baseline, vector((directory / f"in.lj_base_256k_160k_{variant}").read_text()))

  def test_topology_spacing_uses_expression_values_not_variable_names(self):
    content = BASE + "variable length equal 50\nvariable ends atom (v_index%v_length)==0\ngroup ends variable ends\nrun 100"
    self.assertEqual(columns(content)["interaction_settings.topology_spacing.max"], 50)
    self.assertNotEqual(vector(content), vector(content.replace("equal 50", "equal 10")))
    self.assertEqual(vector(content), vector(content.replace("length", "chain_size")))

  def test_hybrid_coefficients_do_not_use_instance_index_as_physics(self):
    values = columns(BASE + """pair_style hybrid/overlay lj/cut 2.5 lj/cut 3.5
pair_coeff 1 1 lj/cut 1 0.5 1
pair_coeff 1 1 lj/cut 2 1.5 2
run 100
""")
    self.assertEqual(values["interaction_settings.coefficient_1.min"], 0.5)
    self.assertEqual(values["interaction_settings.coefficient_1.max"], 1.5)
    self.assertEqual(values["interaction_settings.coefficient_2.max"], 2)

  def test_simultaneous_dumps_increase_output_workload(self):
    single = BASE + "dump a all custom 100 a.dump id x y z\nrun 100"
    double = BASE + "dump a all custom 100 a.dump id x y z\ndump b all custom 100 b.dump id x y z\nrun 100"
    self.assertEqual(columns(double)["output_settings.dump_count.max"], 2)
    self.assertNotEqual(vector(single), vector(double))

  def test_clear_does_not_carry_old_fixes_into_a_new_system(self):
    values = columns(BASE + "clear\nunits metal\natom_style atomic\nrun 100")
    self.assertEqual(values["dynamics_settings.fix.nve.max_count"], 0)

  def test_standard_unit_and_hybrid_atom_style_aliases(self):
    values = columns("variable unit index real\nunits ${unit}\natom_style hybrid full sphere\nrun 0")
    self.assertEqual(values["units.real"], 1)
    for style in ("hybrid", "full", "sphere"):
      self.assertEqual(values[f"atom_style.{style}"], 1)

  def test_all_bundled_decks_have_the_same_finite_schema(self):
    directory = Path(__file__).resolve().parents[3] / "data/input-decks/lammps"
    for path in directory.glob("in.*"):
      with self.subTest(path=path.name):
        values = vector(path.read_text())
        self.assertEqual(len(values), len(DECK_FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in values))
