import unittest

class TestTODO(unittest.TestCase):

  def my_first_test(self) -> None:
    gold = 12
    self.assertEqual(12, gold)
