"""Compare group diameter caps with nearest-threshold-only controls, on CPU.

All statistics are fixed training-cache statistics; no AP or training is used.
Each method evolves its own mapping after the first differing decision.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from simulate_grouping import simulate, validate_matrix


def partition(tasks, groups):
    return [[task for task, value in zip(tasks, groups) if value == group]
            for group in dict.fromkeys(groups)]


def compare(tasks, matrix, order, cap, expand_th=150):
    baseline = simulate(tasks, matrix, order, expand_th, None)
    # Keeping the original nearest condition also prevents a cap > expand_th
    # from relaxing the baseline. Existing strict < is preserved in this control.
    nearest = simulate(tasks, matrix, order, min(expand_th, cap), None)
    diameter = simulate(tasks, matrix, order, expand_th, cap)
    first = next((i for i, (a, b) in enumerate(zip(nearest['groups'], diameter['groups']))
                  if a != b), None)
    first_difference = None
    if first is not None:
        # Histories agree strictly before this task; later changes are not
        # independent direct effects of the diameter gate.
        first_difference = dict(
            task=diameter['order'][first],
            common_prefix_groups=diameter['groups'][:first],
            nearest_control=nearest['records'][first-1],
            diameter_rule=diameter['records'][first-1])
    return dict(cap=cap, baseline=baseline, nearest_control=nearest,
                diameter_rule=diameter, first_difference=first_difference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--caps', nargs='+', type=float,
                        default=[25, 50, 75, 100, 150, 200, 300])
    parser.add_argument('--primary-cap', type=float, default=150)
    parser.add_argument('--expand-th', type=float, default=150)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0
           for v in [args.expand_th, args.primary_cap] + args.caps):
        raise ValueError('All thresholds must be finite and positive')
    if args.primary_cap not in args.caps or len(set(args.caps)) != len(args.caps):
        raise ValueError('Caps must be unique and include the primary cap')
    source_bytes = args.input.read_bytes()
    source = json.loads(source_bytes)
    tasks, matrix = source['tasks'], source['distance_matrix']
    validate_matrix(tasks, matrix)
    domains = list(dict.fromkeys(task.split('_')[0] for task in tasks))
    cases, details = [], []
    for sequence in itertools.permutations(domains):
        order = [i for domain in sequence for i, task in enumerate(tasks)
                 if task.split('_')[0] == domain]
        for cap in args.caps:
            result = compare(tasks, matrix, order, cap, args.expand_th)
            base, near, diam = (result[name] for name in
                                ['baseline', 'nearest_control', 'diameter_rule'])
            cases.append(dict(
                dataset_order='->'.join(sequence), cap=cap,
                baseline_n_groups=base['n_groups'],
                nearest_n_groups=near['n_groups'], diameter_n_groups=diam['n_groups'],
                diameter_differs_from_baseline=diam['groups'] != base['groups'],
                diameter_differs_from_nearest=diam['groups'] != near['groups'],
                diameter_rejections=sum(r['diameter_rejected_original_merge']
                                        for r in diam['records']),
                first_difference_task=(result['first_difference'] or {}).get('task')))
            # Full precision, task-by-task records for the given dataset order.
            if tuple(domains) == sequence:
                for run in [base, near, diam]:
                    run['partition'] = partition(run['order'], run['groups'])
                details.append(result)
    summaries = []
    for cap in args.caps:
        rows = [row for row in cases if row['cap'] == cap]
        summaries.append(dict(
            cap=cap, primary=(cap == args.primary_cap),
            dataset_order_count=len(rows),
            changed_from_baseline=sum(r['diameter_differs_from_baseline'] for r in rows),
            differs_from_nearest_control=sum(r['diameter_differs_from_nearest'] for r in rows),
            baseline_n_groups=sorted(set(r['baseline_n_groups'] for r in rows)),
            nearest_n_groups=sorted(set(r['nearest_n_groups'] for r in rows)),
            diameter_n_groups=sorted(set(r['diameter_n_groups'] for r in rows))))
    result = dict(
        input_sha256=hashlib.sha256(source_bytes).hexdigest(),
        tasks=tasks, original_nearest_threshold=args.expand_th,
        primary_cap=args.primary_cap, exploratory_caps=[v for v in args.caps if v != args.primary_cap],
        cap_grid_fixed_before_execution=True, training_started=False, gpu_used=False,
        methods=dict(baseline='min(new, old) < original_nearest_threshold',
                     nearest_control='min(new, old) < min(original_nearest_threshold, cap)',
                     diameter_rule='original nearest condition AND candidate group diameter <= cap'),
        any_distance_equals_scanned_threshold=any(
            matrix[i][j] == v for i in range(len(tasks)) for j in range(i)
            for v in args.caps + [args.expand_th]),
        summaries=summaries, cases=cases, given_order_details=details,
        limitations=[
            'Six dataset-block orders retain the task order within each domain; these are not training seeds.',
            'Lower caps are exploratory; a forced split below 150 is not a violation of the original 150 threshold.',
            'Changed grouping does not establish AP gain, harmful interference, or an optimal cap.',
            'After the first differing decision each method uses its own mapping history.',
            'The nearest predicate is strict <; diameter is inclusive <=. Report any exact-boundary distances.',
            'No validation/test AP was used for threshold exploration.'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output_dir/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
