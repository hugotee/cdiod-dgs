"""Exercise the actual SVD predictor, without building a detector or using GPUs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    import torch
    import yaml
    from mmengine.config import ConfigDict
    from mmengine.logging import MMLogger
    from projects.DGS.dgs.domain_predictor.svd_domain_predictor import AdaptiveDomainPredictor_SVD
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    MMLogger.get_instance('dtg_cpu_preflight')
    tasks=['DIOR_Task2-1','DIOR_Task2-2','PascalVOC_Task2-1','PascalVOC_Task2-2','RUOD']
    stats=args.baseline_root/'feature_stats/svd'
    assert all((stats/f'{name}.pt').is_file() for name in tasks)

    def run(names,stats_path,cap,name):
        directory=args.output_dir/name
        directory.mkdir(exist_ok=False)
        mapping=directory/'task_id_mapping.yaml'
        mapping.write_text(yaml.safe_dump({0:0}))
        trace=directory/'trace.jsonl'
        for task_id in range(len(names)):
            cfg=ConfigDict(type='svd',multilevel=False,expand_th=150,ood_th=500,
                min_eig_ratio=1e-3,max_group_kl=cap,stats_path=str(stats_path),
                feat_path='UNUSED_ALL_STATS_MUST_EXIST',task_id_mapping_path=str(mapping),
                grouping_trace_path=str(trace))
            AdaptiveDomainPredictor_SVD(cfg,num_tasks=5,moe_modules=[],
                task_id=task_id,seen_tasks=names[:task_id+1],work_dirs=str(directory))
        result=yaml.safe_load(mapping.read_text())
        return [result[i] for i in range(len(names))], [json.loads(v) for v in trace.read_text().splitlines()]

    results=[]
    for cap in [None,150,200,300]:
        groups,trace=run(tasks,stats,cap,'baseline_'+str(cap))
        assert groups==[0,0,1,1,2],groups
        assert len(trace)==4
        results.append(dict(case='stored_five_task_statistics',cap=cap,groups=groups,passed=True))
    synthetic=args.output_dir/'synthetic_stats'
    synthetic.mkdir(exist_ok=False)
    for name,mean in zip(tasks,[0.,10.,-10.]):
        torch.save(dict(mean=torch.tensor([[mean]]),cov=torch.ones(1,1,1)),synthetic/f'{name}.pt')
    for cap,expected in [(None,[0,0,0]),(150,[0,0,1])]:
        groups,trace=run(tasks[:3],synthetic,cap,'synthetic_'+str(cap))
        assert groups==expected,(cap,groups)
        assert abs(trace[-1]['group_diameter_after']-200)<1e-4
        results.append(dict(case='synthetic_gaussian_chain_not_benchmark_evidence',cap=cap,groups=groups,passed=True))
    result=dict(passed=True,cpu_only=True,detector_built=False,training_started=False,cases=results)
    (args.output_dir/'predictor_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
