"""CPU-only grouping simulations using cached train-statistic distances.

No torch, model construction, dataset access, or training is needed here.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/DGS/dgs/domain_predictor'))
from grouping_diagnostics import candidate_group_record, apply_diameter_limit


def validate_matrix(tasks, matrix):
    n = len(tasks)
    if not n or len(set(tasks)) != n or len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError('Expected unique tasks and a square distance matrix')
    for i in range(n):
        for j in range(n):
            a,b = float(matrix[i][j]),float(matrix[j][i])
            if not math.isfinite(a) or a < -1e-4 or not math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-4):
                raise ValueError('Distances must be finite, nonnegative and symmetric')
        if abs(matrix[i][i]) > 1e-4:
            raise ValueError('Expected zero self distances')


def simulate(tasks, matrix, order, expand_th=150, cap=None):
    mapping = {0:0}
    records = []
    for t in range(1,len(order)):
        distances = [matrix[order[t]][order[j]] for j in range(t)]
        pair = lambda a,b: matrix[order[a]][order[b]]
        record = candidate_group_record(t,mapping,distances,pair,expand_th)
        record = apply_diameter_limit(record,cap)
        record['task_name'] = tasks[order[t]]
        record['member_names'] = [tasks[order[j]] for j in record['members_before']]
        records.append(record)
        mapping[t] = record['decided_group']
    return dict(order=[tasks[i] for i in order], max_group_kl=cap,
                groups=[mapping[i] for i in range(len(order))],
                n_groups=len(set(mapping.values())), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--caps',nargs='+',type=float,default=[150,200,300])
    parser.add_argument('--expand-th',type=float,default=150)
    parser.add_argument('--dataset-orders',choices=['given','all'],default='all')
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    tasks,matrix = source['tasks'],source['distance_matrix']
    validate_matrix(tasks,matrix)
    if not math.isfinite(args.expand_th) or args.expand_th <= 0:
        raise ValueError('expand-th must be finite and positive')
    domains = list(dict.fromkeys(t.split('_')[0] for t in tasks))
    domain_orders = list(itertools.permutations(domains)) if args.dataset_orders=='all' else [domains]
    runs = []
    csv_rows = []
    for sequence in domain_orders:
        order = [i for domain in sequence for i,t in enumerate(tasks) if t.split('_')[0]==domain]
        baseline = simulate(tasks,matrix,order,args.expand_th)
        runs.append(baseline)
        for record in baseline['records']:
            csv_rows.append(dict(dataset_order='->'.join(sequence), task=record['task_name'],
                members=';'.join(record['member_names']), min_kl=record['min_kl'],
                max_kl=record['max_kl'], kl_range=record['kl_range'],
                group_diameter_before=record['group_diameter_before'],
                group_diameter_after=record['group_diameter_after'],
                original_accepts=record['original_accepts']))
        for cap in args.caps:
            run = simulate(tasks,matrix,order,args.expand_th,cap)
            run['changed_from_original'] = run['groups'] != baseline['groups']
            run['diameter_rejections'] = sum(r['diameter_rejected_original_merge'] for r in run['records'])
            # Baseline records identify the first intervention before modified
            # group histories diverge. Later decisions use each run's own groups.
            run['baseline_trigger_tasks'] = [r['task_name'] for r in baseline['records']
                if r['original_accepts'] and r['group_diameter_after'] > cap]
            runs.append(run)
    changed = [r for r in runs if r.get('changed_from_original')]
    summary = dict(input=str(args.input),tasks=tasks,expand_th=args.expand_th,
        caps=args.caps,dataset_order_count=len(domain_orders),
        modified_run_count=len(domain_orders)*len(args.caps),
        changed_run_count=len(changed),
        has_observed_trigger=bool(changed),
        training_started=False,
        gate='candidate intervention found; assess paired-run design before training' if changed else
             'no changed grouping at these caps; do not launch an uninformative retraining',
        runs=runs, limitations=['Grouping simulations are not AP experiments.',
            'These dataset-order permutations are not independent training seeds.',
            'No validation/test AP was used to select thresholds.'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'simulation.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output_dir/'baseline_candidate_groups.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
        writer.writeheader();writer.writerows(csv_rows)
    (args.output_dir/'input_matrix.json').write_text(json.dumps(source,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
