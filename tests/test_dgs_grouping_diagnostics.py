"""CPU-only semantic checks, independent of MMDetection and CUDA."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

SOURCE = Path(__file__).resolve().parents[1]/'projects/DGS/dgs/domain_predictor/grouping_diagnostics.py'
spec = importlib.util.spec_from_file_location('grouping_diagnostics', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GroupingDiagnosticsTest(unittest.TestCase):
    def test_singleton_distance_is_not_zero(self):
        row = module.candidate_group_record(1, {0: 0}, [19.14], lambda a,b: 0, 150)
        self.assertEqual(row['kl_range'], 0)
        self.assertEqual(row['group_diameter_after'], 19.14)
        self.assertTrue(row['original_accepts'])

    def test_nearest_neighbor_does_not_bound_whole_group(self):
        row = module.candidate_group_record(2, {0: 0, 1: 0}, [20, 250], lambda a,b: 40, 150)
        self.assertTrue(row['original_accepts'])
        self.assertEqual(row['group_diameter_after'], 250)
        self.assertEqual(row['kl_range'], 230)

    def test_old_old_pair_is_part_of_diameter(self):
        row = module.candidate_group_record(2, {0: 0, 1: 0}, [10, 20], lambda a,b: 300, 150)
        self.assertEqual(row['group_diameter_after'], 300)

    def test_only_nearest_tasks_group_is_candidate(self):
        row = module.candidate_group_record(3, {0: 0, 1: 0, 2: 1}, [400, 300, 25], lambda a,b: 700, 150)
        self.assertEqual(row['members_before'], [2])
        self.assertEqual(row['max_kl'], 25)

    def test_trace_does_not_mutate_mapping(self):
        mapping={0: 0}
        row=module.candidate_group_record(1,mapping,[20],lambda a,b:0,150)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'nested/trace.jsonl'
            module.append_grouping_record(path,row)
            self.assertEqual(json.loads(path.read_text())['group_diameter_after'],20)
        self.assertEqual(mapping,{0: 0})


if __name__ == '__main__':
    unittest.main()
