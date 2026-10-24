"""Run preparation only: predictor checks, cached statistics, and grouping replay."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    repo=args.repo.resolve();out=args.output_dir.resolve()
    # New output directory prevents stale successful evidence from an older run.
    out.mkdir(parents=True,exist_ok=False)
    entry=repo/'tools/dgs_diagnostics'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    state=dict(code_commit=revision,started_at=datetime.now(timezone.utc).isoformat(),
        status='running',training_started=False,gpu_used=False,completed_steps=[])
    def save():
        (out/'preflight_status.json').write_text(json.dumps(state,indent=2)+'\n')
    commands=[
        ('predictor',[str(entry/'check_predictor_cpu.py'),'--repo',str(repo),
            '--baseline-root',str(args.baseline_root.resolve()),'--output-dir',str(out/'predictor')]),
        ('statistics',[str(entry/'prepare_cached_statistics.py'),'--repo',str(repo),
            '--output-dir',str(out/'ten_phase/stats')]),
        ('simulation',[str(entry/'simulate_grouping.py'),'--input',str(out/'ten_phase/stats/distance_matrix.json'),
            '--output-dir',str(out/'ten_phase/simulation'),'--caps','150','200','300','--dataset-orders','all'])]
    save()
    try:
        for name,command in commands:
            state['current_step']=name;save()
            with (out/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable,*command],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['completed_steps'].append(name);save()
        result=json.loads((out/'ten_phase/simulation/simulation.json').read_text())
        state.update(status='complete',has_observed_trigger=result['has_observed_trigger'],
            changed_run_count=result['changed_run_count'],training_gate=result['gate'])
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise
    state['finished_at']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
