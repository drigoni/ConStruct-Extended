import unittest

from rdkit import Chem

from ConStruct.datasets.qm9_dataset import molecule_matches_structure_filter


class QM9StructureFilterTests(unittest.TestCase):
    def setUp(self):
        self.acyclic = Chem.MolFromSmiles("CCCC")
        self.one_ring = Chem.MolFromSmiles("C1CCCCC1")
        self.two_rings = Chem.MolFromSmiles("C1CCC2CCCCC2C1")

    def matches(self, mol, characteristic, mode, value):
        return molecule_matches_structure_filter(mol, characteristic, mode, value)

    def test_ring_count_bounds(self):
        self.assertTrue(self.matches(self.two_rings, "ring_count", "at_least", 2))
        self.assertFalse(self.matches(self.one_ring, "ring_count", "at_least", 2))
        self.assertTrue(self.matches(self.one_ring, "ring_count", "at_most", 1))
        self.assertFalse(self.matches(self.two_rings, "ring_count", "at_most", 1))

    def test_max_cycle_length_bounds(self):
        self.assertTrue(self.matches(self.one_ring, "max_cycle_length", "at_least", 6))
        self.assertFalse(self.matches(self.one_ring, "max_cycle_length", "at_least", 7))
        self.assertTrue(self.matches(self.one_ring, "max_cycle_length", "at_most", 6))
        self.assertFalse(self.matches(self.one_ring, "max_cycle_length", "at_most", 5))
        self.assertTrue(self.matches(self.acyclic, "max_cycle_length", "at_most", 0))

    def test_invalid_filter_configuration(self):
        with self.assertRaises(ValueError):
            self.matches(self.one_ring, "unknown", "at_least", 1)
        with self.assertRaises(ValueError):
            self.matches(self.one_ring, "ring_count", "unknown", 1)
        with self.assertRaises(ValueError):
            self.matches(self.one_ring, "ring_count", "at_least", -1)


if __name__ == "__main__":
    unittest.main()
