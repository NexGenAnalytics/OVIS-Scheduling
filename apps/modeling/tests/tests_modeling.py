import unittest

class Test_1(unittest.TestCase):

  def test_1(self) -> None:
    gold = "test"
    self.assertEqual("test", gold)
