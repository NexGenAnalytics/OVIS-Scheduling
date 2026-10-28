import unittest

class TestTODO(unittest.TestCase):
  def test_todo(self) -> None:
    gold = 12
    self.assertEqual(12, gold)
