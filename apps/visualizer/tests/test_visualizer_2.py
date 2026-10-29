import unittest

from visualizer.metrics import filter_metric

class TestMetrics(unittest.TestCase):

  def test_filter_metric_1(self) -> None:
    rows = [
      {"a": 1, "metric": "A", "c": 3},
      {"a": 10, "metric": "A", "c": 30},
      {"a": 305, "metric": "B", "c": 105}
    ]
    result = filter_metric(rows, "A")

    gold = [
      {"a": 1, "metric": "A", "c": 3},
      {"a": 10, "metric": "A", "c": 30}
    ]
    self.assertEqual(result, gold)

  def test_filter_metric_2(self) -> None:
    result = filter_metric([], "test")
    gold = []
    self.assertEqual(result, gold)

  def test_filter_metric_3(self) -> None:
    rows = [
      {"a": 1, "metric": "A", "c": 3},
      {"a": 10, "metric": "A", "c": 30},
      {"a": 305, "metric": "B", "c": 105}
    ]
    result = filter_metric(rows, "C")

    gold = []
    self.assertEqual(result, gold)
