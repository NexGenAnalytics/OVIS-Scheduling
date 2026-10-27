from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualizer.cli import CPU_METRIC, MEMORY_METRIC, load_profile, main, plot_profile

HEADER = "timestamp,time_rel_s,job_id,component_id,metric,unit,value\n"
SAMPLES = (
  "1020,120,42,7,cpu_cores_used,cores,48\n"
  "1000,0,42,7,mem_active_kb,kB,1048576\n"
  "1010,60,42,7,cpu_cores_used,cores,24\n"
  "1020,120,42,7,mem_active_kb,kB,2097152\n"
)


class TestVisualizer(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = Path(self.directory.name) / "ldms_metrics.csv"
    self.path.write_text(HEADER + SAMPLES)
    self.addCleanup(plt.close, "all")

  def test_sorts_each_resource_without_changing_sample_times(self):
    profile = load_profile(self.path)
    self.assertEqual(profile.series[CPU_METRIC][("42", "7")], [(60, 24), (120, 48)])
    self.assertEqual(profile.series[MEMORY_METRIC][("42", "7")], [(0, 1048576), (120, 2097152)])

  def test_separates_jobs_and_components(self):
    with self.path.open("a") as stream:
      stream.write("1020,120,42,8,cpu_cores_used,cores,8\n1020,120,43,7,cpu_cores_used,cores,16\n")
    profile = load_profile(self.path)
    self.assertEqual(len(profile.series[CPU_METRIC]), 3)
    self.assertEqual(profile.series[CPU_METRIC][("42", "8")], [(120, 8)])

  def test_ignores_unrelated_metrics_and_accepts_optional_columns(self):
    self.path.write_text("\ufefftime_rel_s,metric,value\n2,cpu_cores_used,4\ninvalid,other,invalid\n")
    self.assertEqual(load_profile(self.path).series[CPU_METRIC][("unknown", "unknown")], [(2, 4)])

  def test_invalid_resource_samples_report_the_line(self):
    for value in ("nan", "inf", "-1", "", "invalid"):
      with self.subTest(value=value):
        self.path.write_text(HEADER + f"1000,1,42,7,cpu_cores_used,cores,{value}\n")
        with self.assertRaisesRegex(ValueError, "CSV line 2"):
          load_profile(self.path)

  def test_missing_columns_or_metrics_and_wrong_units_are_rejected(self):
    for text, message in (
      ("metric,value\ncpu_cores_used,48\n", "Missing CSV columns: time_rel_s"),
      (HEADER, "No cpu_cores_used or mem_active_kb"),
      (HEADER + "1000,0,42,7,mem_active_kb,bytes,2048\n", "Unexpected unit"),
    ):
      with self.subTest(message=message):
        self.path.write_text(text)
        with self.assertRaisesRegex(ValueError, message):
          load_profile(self.path)

  def test_two_plots_share_time_and_convert_units(self):
    figure = plot_profile(load_profile(self.path), time_unit="minutes", memory_unit="MiB")
    cpu, memory = figure.axes
    self.assertEqual(list(cpu.lines[0].get_xdata()), [1, 2])
    self.assertEqual(list(cpu.lines[0].get_ydata()), [24, 48])
    self.assertEqual(list(memory.lines[0].get_xdata()), [0, 2])
    self.assertEqual(list(memory.lines[0].get_ydata()), [1024, 2048])
    cpu.set_xlim(0.5, 1.5)
    self.assertEqual(memory.get_xlim(), (0.5, 1.5))

  def test_missing_memory_keeps_an_explanatory_panel(self):
    self.path.write_text(HEADER + "1000,0,42,7,cpu_cores_used,cores,24\n")
    figure = plot_profile(load_profile(self.path))
    self.assertEqual(len(figure.axes), 2)
    self.assertIn("No mem_active_kb samples", figure.axes[1].texts[0].get_text())

  def test_exports_a_png_without_a_gui(self):
    output = Path(self.directory.name) / "plots" / "profile.png"
    with patch("builtins.print"):
      self.assertEqual(main([str(self.path), "--save", str(output), "--no-show"]), 0)
    self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

  def test_file_picker_cancellation_exits_cleanly(self):
    with patch("visualizer.cli.choose_csv", return_value=None):
      self.assertEqual(main([]), 0)
