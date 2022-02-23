import unittest

from .protocols import Boolean, CostType

TEST_BITLEN = 32


class TestBoolean(unittest.TestCase):
    def test_get_cost(self):
        b = Boolean(TEST_BITLEN)
        cost = b.get_cost(b.add())
        self.assertIn(CostType.RT, cost)
        self.assertIn(CostType.MEM, cost)
        self.assertIn(CostType.RT_MEM_PRESSURE, cost)
