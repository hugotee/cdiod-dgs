"""Behavioral checks for the tighter-nearest-threshold control."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/dgs_diagnostics'))
from compare_diameter_controls import compare


class DiameterControlsTest(unittest.TestCase):
    def test_close_nearest_can_chain_even_with_the_same_tighter_threshold(self):
        # All early decisions agree, but C is close to B and far from A.
        result = compare(['A', 'B', 'C'],
                         [[0, 40, 90], [40, 0, 20], [90, 20, 0]], [0, 1, 2], 75)
        self.assertEqual(result['nearest_control']['groups'], [0, 0, 0])
        self.assertEqual(result['diameter_rule']['groups'], [0, 0, 1])
        self.assertEqual(result['first_difference']['task'], 'C')
        self.assertEqual(result['first_difference']['common_prefix_groups'], [0, 0])

    def test_large_cap_must_not_relax_original_nearest_threshold(self):
        result = compare(['A', 'B'], [[0, 175], [175, 0]], [0, 1], 200)
        for method in ['baseline', 'nearest_control', 'diameter_rule']:
            self.assertEqual(result[method]['groups'], [0, 1])
        self.assertIsNone(result['first_difference'])


if __name__ == '__main__':
    unittest.main()
