import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from common.utils import read_file

class TestReadFile(unittest.TestCase):

  def setUp(self) -> None:
    self.temporary_directory = TemporaryDirectory()
    self.addCleanup(self.temporary_directory.cleanup)
    self.path = Path(self.temporary_directory.name) / "values.txt"

  def test_reads_and_converts_non_empty_lines(self):
    content = "10\n\n  20  \n30\n"
    self.path.write_text(content, encoding="utf-8")
    values = read_file(self.path, int)
    gold = [10, 20, 30]
    self.assertEqual(values, gold)

  def test_reads_strings_without_surrounding_whitespace(self):
    content = " first \nsecond\n"
    self.path.write_text(content, encoding="utf-8")
    values = read_file(self.path, str)
    gold = ["first", "second"]
    self.assertEqual(values, gold)
