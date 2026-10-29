import unittest

from visualizer.metrics import max_value

class TestMetrics(unittest.TestCase):

  def test_max_value(self) -> None:
    rows = [12, 100, 1, 4, 8]
    gold = 100
    self.assertEqual(max_value(rows), gold)
