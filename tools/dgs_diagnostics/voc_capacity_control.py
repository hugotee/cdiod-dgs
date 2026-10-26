"""One shared rank32 fourth-stage control; reuse completed rank16 A and B."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from voc_ab import collect, gpu_idle, now, sha256, write_json
from voc_kd_recheck import differences


def prepare(args):
    import torch
    from mmengine.config import Config
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    from tools.dgs_diagnostics.voc_capacity_audit import cpu_check, expand_state, verify_expansion
    root, source = args.root, args.source
    assert not root.exists(), 'A new experiment root is required'
    assert shutil.disk_usage(root.parent).free > 20*2**30
    previous = json.loads((source/'comparison.json').read_text())
    assert previous['A_feature_KD_enabled'] and previous['conditional_stage4_only']
    original = Path(json.loads((source/'protocol.json').read_text())['source_experiment'])
    changed = subprocess.check_output(['git','diff','--name-only','b897608','HEAD',
                                      '--','projects/DGS','mmdet'], text=True).strip()
    assert not changed, changed
    source_config = source/'branches/A_original/train_config.py'
    cfg = Config.fromfile(str(source_config))
    old_cfg = copy.deepcopy(cfg)
    assert cfg.model.moe_cfg.r == 16 and cfg.model.moe_cfg.alpha == 32.
    assert cfg.train_cfg.max_epochs == 12 and cfg.optim_wrapper.optimizer.lr == .0005
    assert cfg.train_dataloader.batch_size == 2 and cfg.optim_wrapper.accumulative_counts == 1
    assert cfg.model.bbox_head.distn_cfg.feat_distn.type == 'inter-class'
    root.mkdir()
    cpu_check(root/'cpu_check.json')
    branch = root/'branches/C_shared_rank32'
    branch.mkdir(parents=True)
    original_base = original/'branches/A_original/diagnostic_states/base.pth'
    raw = torch.load(original_base, map_location='cpu')
    assert sha256(cfg.load_from) == '80010549a499471c6551c06fc4aaa75bc3b362c90960ce8ec3ccc8f8ca4ced5a'
    assert raw['meta']['source_checkpoint'] == cfg.load_from
    expanded = expand_state(raw['state_dict'])
    check = verify_expansion(raw['state_dict'], expanded)
    # Both original teacher/student copies were identical before fourth-stage learning.
    for key in expanded:
        if 'lora_' in key and key.endswith('.3'):
            assert torch.equal(expanded[key], expanded[key[:-1]+'0'])
    checkpoint = root/'expanded_initial_reference.pth'
    torch.save(dict(state_dict=expanded, meta=dict(raw['meta'], capacity_control=True,
               original_base=str(original_base), expansion_seed=314159, rank=32, alpha=64.)), checkpoint)
    write_json(root/'expansion_check.json', check)
    del raw, expanded
    cfg.work_dir = str(branch)
    cfg.load_from = str(checkpoint)
    cfg.model.moe_cfg.r = cfg.moe_cfg.r = 32
    cfg.model.moe_cfg.alpha = cfg.moe_cfg.alpha = 64.
    mapping = branch/'task_id_mapping.yaml'
    shutil.copy2(source/'branches/A_original/task_id_mapping.yaml', mapping)
    cfg.model.domain_predictor_cfg.task_id_mapping_path = str(mapping)
    cfg.model.domain_predictor_cfg.grouping_trace_path = str(branch/'grouping_trace.jsonl')
    cfg.train_dataloader.dataset.distn_cfg.task_id_mapping_path = str(mapping)
    for evaluator in [cfg.val_evaluator, cfg.test_evaluator]:
        evaluator.metrics[0].outfile_prefix = str(branch/'validation_predictions')
    cfg.diagnostic_protocol.update(kind='VOC_stage4_shared_rank32_capacity_control',
        gpu_ids=[2,3], capacity_control=dict(rank=32, alpha=64., final_lora_budget=1376256))
    cfg.custom_imports.imports = ['projects.DGS.dgs', 'tools.dgs_diagnostics.voc_ab_train',
                                 'tools.dgs_diagnostics.voc_capacity_audit']
    cfg.custom_hooks = [h for h in cfg.custom_hooks
                        if h['type'] not in ['VOCFeatureKDAuditHook','VOCDiagnosticAuditHook']]
    cfg.custom_hooks.append(dict(type='VOCDiagnosticAuditHook'))
    audit = dict(type='VOCCapacityKDAuditHook', reference_base=str(checkpoint),
        original_base=str(original_base), reference_branch=str(original/'branches/A_original'),
        preflight=False, audit_steps=32)
    cfg.custom_hooks.append(audit)
    edits = differences(old_cfg.to_dict(), cfg.to_dict())
    allowed = ['work_dir','load_from','moe_cfg.r','moe_cfg.alpha','model.moe_cfg.r','model.moe_cfg.alpha',
        'model.domain_predictor_cfg.task_id_mapping_path','model.domain_predictor_cfg.grouping_trace_path',
        'train_dataloader.dataset.distn_cfg.task_id_mapping_path',
        'val_evaluator.metrics.0.outfile_prefix','test_evaluator.metrics.0.outfile_prefix',
        'diagnostic_protocol','custom_imports','custom_hooks']
    assert all(any(e['path']==p or e['path'].startswith(p+'.') for p in allowed) for e in edits), edits
    write_json(root/'configuration_diff.json', edits)
    config = branch/'train_config.py'
    cfg.dump(str(config))
    for definition, expected_count in [(cfg.train_dataloader.dataset,2147),(cfg.val_dataloader.dataset,3422)]:
        dataset = DATASETS.build(definition)
        assert len(dataset) == expected_count
        del dataset
    preflight = root/'gpu_preflight'
    preflight.mkdir()
    probe = copy.deepcopy(cfg)
    probe.work_dir = str(preflight)
    probe_mapping = preflight/'task_id_mapping.yaml'
    shutil.copy2(mapping, probe_mapping)
    probe.model.domain_predictor_cfg.task_id_mapping_path = str(probe_mapping)
    probe.model.domain_predictor_cfg.grouping_trace_path = str(preflight/'grouping_trace.jsonl')
    probe.train_dataloader.dataset.distn_cfg.task_id_mapping_path = str(probe_mapping)
    probe.custom_hooks = [copy.deepcopy(h) for h in cfg.custom_hooks if h['type']=='WeightsTransformHook']
    probe.custom_hooks.append(dict(audit, preflight=True))
    probe.default_hooks.checkpoint = None
    probe.default_hooks.logger.interval = 8
    probe.train_cfg = dict(type='IterBasedTrainLoop',max_iters=32,val_interval=1000000)
    probe.val_dataloader = probe.val_cfg = probe.val_evaluator = None
    probe.test_dataloader = probe.test_cfg = probe.test_evaluator = None
    probe.param_scheduler = None
    probe_config = preflight/'train_config.py'
    probe.dump(str(probe_config))
    old_protocol = json.loads((source/'protocol.json').read_text())
    inputs = dict(old_protocol['input_sha256'])
    for path in [checkpoint,original_base,source_config,source/'comparison.json',
                 Path(previous['A']['checkpoint']),Path(previous['B']['checkpoint'])]:
        inputs[str(path)] = sha256(path)
    protocol = dict(created_at=now(),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),source_experiment=str(source),
        original_experiment=str(original),tasks=old_protocol['tasks'],seed=old_protocol['seed'],
        gpu_ids=[2,3],per_gpu_batch=2,effective_batch=4,accumulation=1,epochs=12,
        branch='C_shared_rank32',config=str(config),config_sha256=sha256(config),
        preflight_config=str(probe_config),preflight_config_sha256=sha256(probe_config),
        input_sha256=inputs,final_lora_parameters=dict(A=688128,B=1376256,C=1376256),
        reused_results=dict(A=previous['A'],B=previous['B']),training_started=False,
        original_prefix_sha256='80010549a499471c6551c06fc4aaa75bc3b362c90960ce8ec3ccc8f8ca4ced5a',
        limitations=['Single-seed conditional fourth-stage capacity screen, not a full corrected reproduction.',
            'Matches final LoRA budget, not trainable or total parameters; B has extra65536 DN parameters.',
            'B freezes its old group; C trains a larger shared student. Branch initialization and merge differ.',
            'A/B are reused results, with first32 augmented inputs checked on both ranks.',
            'A similar C score does not prove capacity is the sole cause; a B lead is not a novelty claim.'])
    write_json(root/'protocol.json',protocol)
    write_json(root/'status.json',dict(stage='prepared',updated_at=now(),training_started=False,gpu_ids=[2,3]))
    print(json.dumps(dict(root=str(root),CPU_passed=True,expansion=check,train_jobs=1,epochs=12)))


def run(args):
    root = args.root
    lock = (root/'driver.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = json.loads((root/'protocol.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==protocol['source_commit']
    for path, digest in protocol['input_sha256'].items():
        assert sha256(path)==digest, path
    child = None
    def status(stage, **extra):
        write_json(root/'status.json',dict(stage=stage,updated_at=now(),driver_pid=os.getpid(),
            gpu_ids=[2,3],**extra))
    def terminate(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        status('interrupted',signal=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    for preflight in [True,False]:
        config = protocol['preflight_config' if preflight else 'config']
        digest = protocol['preflight_config_sha256' if preflight else 'config_sha256']
        assert sha256(config)==digest
        directory = Path(config).parent
        assert not list(directory.glob('epoch_*.pth')), 'Refusing an implicit restart over existing checkpoints'
        for gpu in [2,3]: gpu_idle(gpu)
        command = [sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',
            f'--master_port={args.port+int(preflight)}',str(Path(__file__).resolve()),'worker',
            '--repo',str(args.repo),'--root',str(root),'--source',str(args.source),'--config',config]
        log = directory/('train.log' if preflight else 'train.dual.log')
        with log.open('a') as stream:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
            status('gpu_preflight' if preflight else 'train',child_pid=child.pid,
                training_started=not preflight,disposable_preflight_training=preflight,
                phase_dir=str(directory),config=config,log=str(log),name='C_shared_rank32')
            code = child.wait()
        child = None
        assert code == 0, f'Training process failed: {log}'
        rows = [json.loads((directory/f'capacity_audit_rank{rank}.json').read_text()) for rank in range(2)]
        assert all(row['passed'] for row in rows)
        if preflight:
            assert not list(directory.glob('*.pth'))
            write_json(root/'setup_check_dual.json',dict(passed=True,ranks=rows,
                function_preserving_initialization=True,extra_capacity_learns=True,
                first32_inputs_paired=True,feature_KD_active=True,
                temporary_weights_not_saved=True,formal_restarts_from_expanded_initial_reference=True))
    branch = root/'branches/C_shared_rank32'
    c = collect(branch)
    write_json(branch/'stage_complete.json',c)
    results = dict(protocol['reused_results'],C=c)
    pairing = {}
    original = Path(protocol['original_experiment'])/'branches/A_original'
    for rank in range(2):
        read = lambda p: json.loads(p.read_text())['sha256']
        pairing[str(rank)] = read(branch/f'first32_input_fingerprint_rank{rank}.json') == read(
            original/f'first32_input_fingerprint_rank{rank}.json')
    assert all(pairing.values())
    for path, digest in protocol['input_sha256'].items():
        assert sha256(path)==digest, f'Input changed: {path}'
    results.update(single_seed=True,conditional_stage4_only=True,rank_input_pairing=pairing,
        final_lora_parameters=protocol['final_lora_parameters'],exact_total_parameter_match=False,
        delta_AP_points={name:{metric:100*(results[x][metric]-results[y][metric])
            for metric in ['old15_ap','new5_ap','all20_ap']}
            for name,x,y in [('C_minus_A','C','A'),('B_minus_C','B','C')]})
    write_json(root/'comparison.json',results)
    status('complete',training_started=True,comparison=str(root/'comparison.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','launch','worker'])
    for name in ['repo','root','source']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config')
    parser.add_argument('--port',type=int,default=29594)
    args = parser.parse_args()
    args.repo,args.root,args.source = [p.resolve() for p in [args.repo,args.root,args.source]]
    os.chdir(args.repo)
    sys.path.insert(0,str(args.repo))
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.mode=='prepare' else '2,3',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(args.repo),
        TMPDIR=str(args.repo.parent/'tmp'),XDG_CACHE_HOME=str(args.repo.parent/'.cache'),
        TORCH_HOME=str(args.repo.parent/'.cache/torch'),HF_HOME=str(args.repo.parent/'.cache/huggingface'))
    if args.mode=='prepare':
        prepare(args)
    elif args.mode=='worker':
        import torch
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        import projects.DGS.dgs
        torch.set_num_threads(4)
        cfg = Config.fromfile(args.config)
        cfg.launcher = 'pytorch'
        Runner.from_cfg(cfg).train()
    elif args.mode=='launch':
        for gpu in [2,3]: gpu_idle(gpu)
        command = [sys.executable,str(Path(__file__).resolve()),'run','--repo',str(args.repo),
            '--root',str(args.root),'--source',str(args.source),'--port',str(args.port)]
        with (args.root/'driver.log').open('a') as stream:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(driver_pid=child.pid,root=str(args.root),gpu_ids=[2,3])))
    else:
        try:
            run(args)
        except Exception as exc:
            status = json.loads((args.root/'status.json').read_text())
            status.update(stage='failed',updated_at=now(),error=str(exc))
            write_json(args.root/'status.json',status)
            raise


if __name__=='__main__':
    main()
