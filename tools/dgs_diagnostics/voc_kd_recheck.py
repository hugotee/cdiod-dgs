"""Bounded stage-four feature-KD recheck on GPUs2,3, reusing unaffected B.

Only a new experimental config enables the existing inter-class implementation.
The first three trained tasks, model code, loss code, and old artifacts stay intact.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from voc_ab import gpu_idle, now, sha256, write_json


def differences(old, new, path=''):
    result = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            name = f'{path}.{key}' if path else key
            if key not in old or key not in new:
                result.append(dict(path=name, before=old.get(key), after=new.get(key)))
            else:
                result.extend(differences(old[key], new[key], name))
    elif isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        for index in range(max(len(old), len(new))):
            name = f'{path}.{index}'
            if index >= len(old) or index >= len(new):
                result.append(dict(path=name, before=old[index] if index < len(old) else None,
                                   after=new[index] if index < len(new) else None))
            else:
                result.extend(differences(old[index], new[index], name))
    elif old != new:
        result.append(dict(path=path, before=old, after=new))
    return result


def prepare(args):
    from mmengine.config import Config
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    root, source = args.root, args.source
    if root.exists():
        raise FileExistsError('Preparation requires a new experiment root')
    cpu = json.loads(args.cpu_check.read_text())
    assert cpu['passed'] and cpu['total_loss_includes_both']
    protocol_old = json.loads((source/'protocol.json').read_text())
    a_source = source/'branches/A_original'
    b_source = source/'branches/B_diameter75'
    old = Config.fromfile(str(a_source/'train_config.py'))
    b = Config.fromfile(str(b_source/'train_config.py'))
    assert old.model.type == 'GroundingDINO_DGS'
    assert b.model.type == 'GroundingDINO_DGS_Base'
    assert 'distn_cfg' not in b.model and 'distn_cfg' not in b.model.bbox_head
    assert old.train_dataloader.batch_size == 2 and old.optim_wrapper.accumulative_counts == 1
    assert old.train_cfg.max_epochs == 12 and old.optim_wrapper.optimizer.lr == .0005
    changed = subprocess.check_output(['git','diff','--name-only',
        protocol_old['source_commit'], 'HEAD','--','projects/DGS','mmdet'],text=True).strip()
    assert not changed, f'Model or loss source changed since reused B: {changed}'
    assert shutil.disk_usage(root.parent).free > 20*2**30
    root.mkdir()
    shutil.copy2(args.cpu_check, root/'cpu_check.json')
    a = root/'branches/A_original'
    a.mkdir(parents=True)
    mapping = a/'task_id_mapping.yaml'
    shutil.copy2(a_source/'task_id_mapping.yaml', mapping)
    cfg = copy.deepcopy(old)
    cfg.work_dir = str(a)
    cfg.model.domain_predictor_cfg.task_id_mapping_path = str(mapping)
    cfg.model.domain_predictor_cfg.grouping_trace_path = str(a/'grouping_trace.jsonl')
    cfg.train_dataloader.dataset.distn_cfg.task_id_mapping_path = str(mapping)
    paths = ['distn_cfg', 'model.distn_cfg', 'model.bbox_head.distn_cfg',
             'train_dataloader.dataset.distn_cfg']
    for path in paths:
        item = cfg
        for part in path.split('.'):
            item = item[part]
        assert item.feat_distn.type == 'inter-intra'
        item.feat_distn.type = 'inter-class'
    for evaluator in [cfg.val_evaluator, cfg.test_evaluator]:
        evaluator.metrics[0].outfile_prefix = str(a/'validation_predictions')
    cfg.diagnostic_protocol.update(kind='VOC_stage4_feature_KD_recheck', gpu_ids=[2,3],
        feature_distillation_configuration='inter-class enables existing topology KD, weights3/5')
    cfg.custom_imports.imports.append('tools.dgs_diagnostics.voc_kd_audit')
    hook = dict(type='VOCFeatureKDAuditHook', reference_base=str(a_source/'diagnostic_states/base.pth'),
                reference_branch=str(a_source), preflight=False, audit_steps=32)
    cfg.custom_hooks.append(hook)
    edits = differences(old.to_dict(), cfg.to_dict())
    allowed = ['work_dir','model.domain_predictor_cfg.task_id_mapping_path',
        'model.domain_predictor_cfg.grouping_trace_path',
        'train_dataloader.dataset.distn_cfg.task_id_mapping_path',
        'val_evaluator.metrics.0.outfile_prefix','test_evaluator.metrics.0.outfile_prefix',
        'diagnostic_protocol','custom_imports','custom_hooks'] + [p+'.feat_distn.type' for p in paths]
    assert all(any(row['path']==p or row['path'].startswith(p+'.') for p in allowed) for row in edits), edits
    config = a/'train_config.py'
    cfg.dump(str(config))
    write_json(root/'configuration_diff.json', edits)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    assert len(dataset) == 2147 and len(dataset.metainfo['classes']) == 20
    del dataset
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    assert len(dataset) == 3422 and len(dataset.datasets[0].metainfo['classes']) == 20
    del dataset
    # B did not run the consolidation detector/head or feature-KD path. No model
    # code changes are introduced; retain its exact original results and pairing.
    b_target = root/'branches/B_diameter75'
    b_target.mkdir()
    for name in ['stage_complete.json','branch_origin.json','first32_input_fingerprint.json',
                 'first32_input_fingerprint_rank0.json','first32_input_fingerprint_rank1.json']:
        shutil.copy2(b_source/name, b_target/name)
    write_json(b_target/'reused_result.json', dict(reused=True, source=str(b_source),
        checkpoint=str(b_source/'epoch_12.pth'), checkpoint_sha256=sha256(b_source/'epoch_12.pth'),
        model_type=b.model.type, no_distillation_config_in_model=True,
        model_loss_source_unchanged_since_training=True))
    # Short disposable DDP training: verify actual data/loss/backward on both
    # ranks, then restart production from the untouched common prefix.
    preflight = root/'gpu_preflight'
    preflight.mkdir()
    probe = copy.deepcopy(cfg)
    probe.work_dir = str(preflight)
    probe_mapping = preflight/'task_id_mapping.yaml'
    shutil.copy2(mapping, probe_mapping)
    probe.model.domain_predictor_cfg.task_id_mapping_path = str(probe_mapping)
    probe.model.domain_predictor_cfg.grouping_trace_path = str(preflight/'grouping_trace.jsonl')
    probe.train_dataloader.dataset.distn_cfg.task_id_mapping_path = str(probe_mapping)
    probe.custom_imports.imports = ['projects.DGS.dgs']
    probe.custom_hooks = [copy.deepcopy(h) for h in cfg.custom_hooks if h['type'] == 'WeightsTransformHook']
    probe.custom_hooks.append(dict(hook, preflight=True))
    probe.default_hooks.pop('checkpoint', None)
    probe.default_hooks.logger.interval = 8
    probe.train_cfg = dict(type='IterBasedTrainLoop', max_iters=32, val_interval=1000000)
    probe.val_dataloader = probe.val_cfg = probe.val_evaluator = None
    probe.test_dataloader = probe.test_cfg = probe.test_evaluator = None
    probe.param_scheduler = None
    probe_config = preflight/'train_config.py'
    probe.dump(str(probe_config))
    plan = copy.deepcopy(next(p for p in protocol_old['plans'] if p['name']=='A_original'))
    plan.update(directory=str(a),config=str(config),config_sha256=sha256(config))
    inputs = dict(protocol_old['input_sha256'])
    for p in [Path(cfg.load_from), a_source/'diagnostic_states/base.pth',
              b_source/'epoch_12.pth', a_source/'train_config.py', b_source/'train_config.py']:
        inputs[str(p)] = sha256(p)
    protocol = dict(created_at=now(), source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(), source_experiment=str(source),
        tasks=protocol_old['tasks'],seed=protocol_old['seed'],gpu_ids=[2,3],effective_batch=4,
        microbatch=2,accumulation=1,plans=[plan],input_sha256=inputs,
        preflight_config=str(probe_config),preflight_config_sha256=sha256(probe_config),
        original_A_metrics=str(a_source/'stage_complete.json'),
        reused_B_metrics=str(b_source/'stage_complete.json'),training_started=False,
        B_reused=True, model_and_loss_code_unchanged=True,
        changed_feature_types=paths,
        limitations=['Conditional fourth-stage experiment: the first three tasks retain their original training.',
            'Enables the existing official-code topology implementation; not a claim of exact paper reproduction.',
            'Sparse pseudo-class batches can skip topology terms under the existing implementation.',
            'B is an unaffected new-group branch; only A is retrained.',
            'One seed and order; added-group capacity and branch-policy confounds remain.',
            'GPU allocation changes1,3 to2,3; deterministic input pairing is checked, CUDA bitwise identity is not promised.'])
    write_json(root/'protocol.json',protocol)
    write_json(root/'status.json',dict(stage='prepared',updated_at=now(),training_started=False,gpu_ids=[2,3]))
    print(json.dumps(dict(root=str(root),train_jobs=1,reused_B=True,preflight_iterations=32,
                         epochs=12,nominal_effective_batch=4),indent=2))


def run(args):
    from types import SimpleNamespace
    from voc_ab import run as train_run
    lock=(args.root/'bootstrap.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((args.root/'protocol.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==protocol['source_commit']
    assert sha256(protocol['preflight_config'])==protocol['preflight_config_sha256']
    for path, expected in protocol['input_sha256'].items():
        assert sha256(path)==expected, path
    for gpu in [2,3]:
        gpu_idle(gpu)
    command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',
             f'--master_port={args.port+1}', str(args.repo/'tools/dgs_diagnostics/voc_kd_audit.py'),
             '--preflight-config',protocol['preflight_config']]
    log=args.root/'gpu_preflight/train.log'
    with log.open('a') as stream:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                               start_new_session=True)
        write_json(args.root/'status.json',dict(stage='gpu_preflight',updated_at=now(),
            driver_pid=os.getpid(),child_pid=child.pid,gpu_ids=[2,3],training_started=False,
            disposable_preflight_training=True,log=str(log),phase_dir=str(log.parent)))
        if child.wait():
            raise RuntimeError(f'GPU preflight failed; inspect {log}')
    rows=[json.loads((log.parent/f'feature_kd_audit_rank{rank}.json').read_text()) for rank in range(2)]
    assert all(row['passed'] for row in rows)
    write_json(args.root/'setup_check_dual.json',dict(passed=True,gpu_preflight=True,world_size=2,
        nominal_effective_batch=4,actual_nonzero_image_and_text_KD=True,
        actual_KD_backward_and_feature_gradients=True,initial_lora_matches_original_A=True,
        both_rank_input_pairing=True,student_updated_teacher_unchanged=True,
        disposable_weights_discarded=True,ranks=rows))
    train_run(SimpleNamespace(root=args.root,repo=args.repo,stats=args.source/'frozen_stats',
                              gpus=[2,3],port=args.port))
    comparison=json.loads((args.root/'comparison.json').read_text())
    original=json.loads(Path(protocol['original_A_metrics']).read_text())
    comparison.update(A_feature_KD_enabled=True,B_reused=True,conditional_stage4_only=True,
        delta_AP_points_A_KD_minus_original_A={key:100*(comparison['A'][key]-original[key])
            for key in ['old15_ap','new5_ap','all20_ap']})
    write_json(args.root/'comparison.json',comparison)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','launch'])
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--cpu-check',type=Path)
    parser.add_argument('--port',type=int,default=29586)
    args=parser.parse_args()
    args.repo,args.root,args.source=[p.resolve() for p in [args.repo,args.root,args.source]]
    os.chdir(args.repo)
    sys.path.insert(0,str(args.repo))
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.mode=='prepare' else '2,3',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(args.repo),
        TMPDIR=str(args.repo.parent/'tmp'),XDG_CACHE_HOME=str(args.repo.parent/'.cache'),
        TORCH_HOME=str(args.repo.parent/'.cache/torch'),HF_HOME=str(args.repo.parent/'.cache/huggingface'))
    if args.mode=='prepare':
        prepare(args)
    elif args.mode=='launch':
        for gpu in [2,3]: gpu_idle(gpu)
        command=[sys.executable,str(Path(__file__).resolve()),'run','--repo',str(args.repo),
            '--root',str(args.root),'--source',str(args.source),'--port',str(args.port)]
        with (args.root/'driver.log').open('a') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   start_new_session=True)
        print(json.dumps(dict(driver_pid=child.pid,root=str(args.root),gpu_ids=[2,3])))
    else:
        try:
            run(args)
        except Exception as exc:
            status=json.loads((args.root/'status.json').read_text())
            status.update(stage='failed',updated_at=now(),error=str(exc))
            write_json(args.root/'status.json',status)
            raise


if __name__=='__main__':
    main()
