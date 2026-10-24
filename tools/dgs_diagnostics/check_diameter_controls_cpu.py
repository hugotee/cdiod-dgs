"""Check offline controls against the actual SVD predictor, without a detector."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

from compare_diameter_controls import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cap', type=float, default=75)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    import yaml
    from mmengine.config import ConfigDict
    from mmengine.logging import MMLogger
    from projects.DGS.dgs.domain_predictor.svd_domain_predictor import AdaptiveDomainPredictor_SVD
    torch.set_num_threads(4)
    MMLogger.get_instance('dtg_diameter_controls_cpu')
    source = json.loads((args.stats/'distance_matrix.json').read_text())
    tasks = source['tasks']
    if not all((args.stats/f'{task}.pt').is_file() for task in tasks):
        raise FileNotFoundError('Every cached statistic must exist; GPU extraction is disabled')
    expected = compare(tasks, source['distance_matrix'], list(range(len(tasks))), args.cap)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, threshold, cap, offline in [
            ('baseline', 150, None, expected['baseline']),
            ('nearest_control', min(150, args.cap), None, expected['nearest_control']),
            ('diameter_rule', 150, args.cap, expected['diameter_rule'])]:
        directory = args.output_dir/name
        directory.mkdir()
        mapping = directory/'task_id_mapping.yaml'
        mapping.write_text(yaml.safe_dump({0: 0}))
        trace_path = directory/'trace.jsonl'
        for task_id in range(len(tasks)):
            cfg = ConfigDict(type='svd', multilevel=False, expand_th=threshold,
                             ood_th=500, min_eig_ratio=1e-3, max_group_kl=cap,
                             stats_path=str(args.stats), feat_path='UNUSED_ALL_STATS_EXIST',
                             task_id_mapping_path=str(mapping), grouping_trace_path=str(trace_path))
            AdaptiveDomainPredictor_SVD(cfg, num_tasks=len(tasks), moe_modules=[],
                                       task_id=task_id, seen_tasks=tasks[:task_id+1],
                                       work_dirs=str(directory))
        mapping_values = yaml.safe_load(mapping.read_text())
        groups = [mapping_values[i] for i in range(len(tasks))]
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        if groups != offline['groups'] or len(trace) != len(offline['records']):
            raise AssertionError((name, groups, offline['groups']))
        error = max(abs(a[key]-b[key]) for a, b in zip(trace, offline['records'])
                    for key in ['min_kl', 'group_diameter_before', 'group_diameter_after'])
        if error > 1e-3:
            raise AssertionError((name, error))
        cases.append(dict(method=name, nearest_threshold=threshold, cap=cap,
                          groups=groups, max_distance_error=error, passed=True))
    result = dict(passed=True, cpu_only=True, detector_built=False, training_started=False,
                  source_commit=subprocess.check_output(
                      ['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  cases=cases)
    (args.output_dir/'predictor_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
