"""Check real saved structures, untouched LoRA loading, merge math and routing."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    os.chdir(args.repo);sys.path.insert(0,str(args.repo))
    import gc
    import torch
    from mmengine.config import Config
    from mmengine.logging import MMLogger
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    from voc_inference_eval import lora_digest,configure_router
    torch.set_num_threads(4)
    MMLogger.get_instance('voc_inference_cpu')
    protocol=json.loads((args.root/'protocol.json').read_text())
    jobs={job['case']:job for job in protocol['jobs']}
    states={name:torch.load(jobs[name]['checkpoint'],map_location='cpu')['state_dict']
            for name in ['A_base_fixed0','A_new_fixed0','A_merged_normal']}
    base,new,merged=[states[name] for name in ['A_base_fixed0','A_new_fixed0','A_merged_normal']]
    old_keys=[key for key in base if 'lora_' in key and key.endswith('.0')]
    assert len(old_keys)==48
    assert all(torch.equal(base[key],base[key.rsplit('.',1)[0]+'.3']) for key in old_keys)
    assert all(torch.equal(base[key],new[key]) for key in old_keys)
    assert any(not torch.equal(new[key],new[key.rsplit('.',1)[0]+'.3']) for key in old_keys)
    for key in old_keys:
        expected=.2*new[key]+.8*new[key.rsplit('.',1)[0]+'.3']
        assert torch.allclose(expected,merged[key],rtol=1e-5,atol=1e-7),key
    rows=[]
    for name in ['A_new_fixed0','A_merged_normal','B_normal']:
        job=jobs[name];cfg=Config.fromfile(job['config'])
        model=MODELS.build(cfg.model)
        state=states[name] if name in states else torch.load(job['checkpoint'],map_location='cpu')['state_dict']
        model.load_state_dict(state,strict=True)
        assert lora_digest(model.state_dict())==lora_digest(state)
        parameter_ids=[[list(expert.lora_B.keys()) for expert in module.experts]
                       for module in model.moe_modules if hasattr(module,'experts')]
        expected_ids={'A_new_fixed0':[['0','3']],'A_merged_normal':[['0']],
                      'B_normal':[['0'],['3']]}[name]
        assert parameter_ids and all(value==expected_ids for value in parameter_ids)
        if name=='B_normal':
            for group,task in [(0,0),(1,3)]:
                returned=configure_router(model,group,[])
                assert returned==task
                pred=model.domain_predictor(None,input_sample=None)
                assert pred==task
                model.set_moe_task_id(pred)
                assert all(module.val_task_id==group for module in model.moe_modules)
        rows.append(dict(case=name,strict_load=True,parameter_ids=expected_ids,passed=True))
        del model,state;gc.collect()
    result=dict(passed=True,cpu_only=True,training_started=False,
        base_student_equals_old=True,premerge_old_unchanged=True,premerge_student_trained=True,
        actual_merge_matches_saved_parameters=True,forced_group_to_task_mapping_verified=True,cases=rows)
    (args.root/'cpu_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
