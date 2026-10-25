"""Bounded six-pass inference diagnostic on the completed VOC A/B weights."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from voc_ab import TASKS, now, sha256, write_json, gpu_idle


def ap_summary(metrics):
    values = [metrics[f'PascalVOC/{task}/coco/bbox_mAP'] for task in TASKS]
    return dict(old15_ap=sum(values[:3])/3, new5_ap=values[3],
                all20_ap=sum(values)/4, per_task_ap=dict(zip(TASKS, values)))


def prepare(args):
    from mmengine.config import Config
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    root, source = args.root, args.source
    root.mkdir(parents=True, exist_ok=False)
    definitions = [
        ('A_merged_normal', 'A_original', 'epoch_12.pth', None, True),
        ('B_normal', 'B_diameter75', 'epoch_12.pth', None, True),
        ('A_base_fixed0', 'A_original', 'diagnostic_states/base.pth', 0, False),
        ('A_new_fixed0', 'A_original', 'diagnostic_states/new_premerge.pth', 0, False),
        ('A_merged_fixed0', 'A_original', 'epoch_12.pth', 0, True),
        ('B_old_fixed0', 'B_diameter75', 'epoch_12.pth', 0, True)]
    jobs = []
    inputs = {}
    for name, branch, checkpoint_name, group, merged in definitions:
        directory = root/name
        directory.mkdir()
        checkpoint = source/'branches'/branch/checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        mapping = directory/'task_id_mapping.yaml'
        shutil.copy2(source/'branches'/branch/'task_id_mapping.yaml', mapping)
        cfg = Config.fromfile(str(source/'branches'/branch/'train_config.py'))
        cfg.work_dir = str(directory)
        cfg.load_from = str(checkpoint)
        cfg.resume = False
        cfg.train_dataloader = cfg.train_cfg = cfg.optim_wrapper = cfg.param_scheduler = None
        cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
        cfg.custom_hooks = [dict(type='VOCInferenceAuditHook')]
        cfg.default_hooks.pop('checkpoint', None)
        cfg.default_hooks.logger.interval = 100
        cfg.model_wrapper_cfg = dict(type='MMDistributedDataParallel', broadcast_buffers=False,
                                     find_unused_parameters=False)
        cfg.model.domain_predictor_cfg.task_id_mapping_path = str(mapping)
        cfg.model.domain_predictor_cfg.grouping_trace_path = str(directory/'grouping_trace.jsonl')
        if branch == 'A_original' and merged:
            # LoraLinearRep's ema mode constructs a temporary current expert.
            # A merged checkpoint contains only the group's base expert. The
            # 'none' construction creates exactly that saved structure; no
            # forward equations or checkpoint tensor values are changed.
            cfg.model.moe_cfg.group_cfg.merge_method = 'none'
        cfg.test_dataloader.update(batch_size=1, num_workers=2, persistent_workers=True)
        cfg.test_cfg = dict(type='TestLoop', fp16=False)
        cfg.test_evaluator.metrics[0].outfile_prefix = str(directory/'predictions')
        cfg.test_evaluator.evaluation_tasks = ','.join(TASKS)
        cfg.inference_diagnostic = dict(case=name, forced_group=group,
            diagnostic_task_information=(group is not None), training_started=False)
        config = directory/'test_config.py'
        cfg.dump(str(config))
        inputs.setdefault(str(checkpoint), sha256(checkpoint))
        jobs.append(dict(case=name, branch=branch, checkpoint=str(checkpoint),
            checkpoint_sha256=inputs[str(checkpoint)], config=str(config),
            config_sha256=sha256(config), forced_group=group, merged_structure=merged))
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    assert len(dataset) == 3422 and len(dataset.datasets[0].metainfo['classes']) == 20
    annotation = args.repo/'data/CDIOD/PascalVOC/valid/annotations_without_background.json'
    inputs[str(annotation)] = sha256(annotation)
    protocol = dict(created_at=now(), source_experiment=str(source), gpu_ids=[2, 3],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        jobs=jobs, input_sha256=inputs, validation_images=len(dataset), prompt_classes=20,
        fp16=False, per_gpu_batch=1, training_started=False,
        reference_metrics=str(source/'comparison.json'),
        limits=['Forced groups use task information and are diagnostic references, not deployment scores.',
                'Fixed group is not guaranteed to be the best group.',
                'No optimizer, backward, KD repair, parameter merge or group initialization is executed.',
                'Normal A/B references must reproduce the previous AP within0.05 points before other passes.'])
    write_json(root/'protocol.json', protocol)
    write_json(root/'status.json', dict(stage='prepared', updated_at=now(), training_started=False))
    print(json.dumps(dict(root=str(root), cases=[job['case'] for job in jobs],
                         validation_images=3422, training_started=False), indent=2))


def run(args):
    root = args.root
    lock = (root/'driver.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((root/'protocol.json').read_text())
    if not json.loads((root/'cpu_check.json').read_text())['passed']:
        raise RuntimeError('CPU structure/load/route checks must pass')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != protocol['source_commit']:
        raise RuntimeError('Prepared source commit differs')
    for path, value in protocol['input_sha256'].items():
        if sha256(path) != value:
            raise RuntimeError(f'Source input changed: {path}')
    reference = json.loads(Path(protocol['reference_metrics']).read_text())
    child = None
    def status(stage, **kwargs):
        write_json(root/'status.json', dict(stage=stage, updated_at=now(), driver_pid=os.getpid(),
            training_started=False, gpu_ids=[2, 3], **kwargs))
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        status('interrupted', signal=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    results = {}
    for index, job in enumerate(protocol['jobs']):
        directory = root/job['case']
        finished = directory/'result.json'
        if finished.exists():
            results[job['case']] = json.loads(finished.read_text())
            continue
        if sha256(job['config']) != job['config_sha256']:
            raise RuntimeError('Prepared config changed')
        for gpu in [2, 3]:
            gpu_idle(gpu)
        command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
            f'--master_port={args.port}', str(args.repo/'tools/dgs_diagnostics/voc_inference_eval.py'),
            job['config']]
        log = directory/'evaluate.log'
        status('inference', case=job['case'], pass_number=index+1, total_passes=len(protocol['jobs']),
               phase_dir=str(directory), log=str(log))
        started = time.monotonic()
        with log.open('a') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            status('inference', case=job['case'], pass_number=index+1, total_passes=len(protocol['jobs']),
                   phase_dir=str(directory), log=str(log), child_pid=child.pid)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"{job['case']} exited with code {code}; inspect {log}")
        metrics = json.loads((directory/'metrics.json').read_text())
        summary = ap_summary(metrics)
        counts = Counter()
        for rank in range(2):
            routes = json.loads((directory/f'routes_rank{rank}.json').read_text())
            assert routes['lora_unchanged']
            counts.update({int(key):value for key, value in routes['counts'].items()})
        if sum(counts.values()) != protocol['validation_images']:
            raise RuntimeError('Router audit did not cover the complete validation set')
        if job['forced_group'] is not None and dict(counts) != {job['forced_group']:3422}:
            raise RuntimeError('Forced group was not consistently selected')
        check = None
        if job['case'] in ['A_merged_normal', 'B_normal']:
            original = reference['A' if job['case'].startswith('A_') else 'B']
            error = max(abs(summary[key]-original[key]) for key in ['old15_ap','new5_ap','all20_ap'])
            check = dict(passed=error <= .0005, max_ap_error=error)
            if not check['passed']:
                raise RuntimeError(f'Normal-route reference failed: {check}')
        result = dict(**summary, route_counts=dict(counts), reference_check=check,
                      elapsed_seconds=time.monotonic()-started, forced_group=job['forced_group'],
                      diagnostic_reference=job['forced_group'] is not None, training_started=False)
        write_json(finished, result)
        results[job['case']] = result
    write_json(root/'comparison.json', dict(results=results, training_started=False,
        A_training_delta_AP={key:100*(results['A_new_fixed0'][key]-results['A_base_fixed0'][key])
            for key in ['old15_ap','new5_ap','all20_ap']},
        A_merge_delta_AP={key:100*(results['A_merged_fixed0'][key]-results['A_new_fixed0'][key])
            for key in ['old15_ap','new5_ap','all20_ap']},
        B_old_fixed_minus_normal_AP=100*(results['B_old_fixed0']['old15_ap']-results['B_normal']['old15_ap']),
        forced_routing_is_task_informed_diagnostic=True))
    status('complete', comparison=str(root/'comparison.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'launch'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--port', type=int, default=29585)
    args = parser.parse_args()
    args.repo, args.root, args.source = [path.resolve() for path in [args.repo,args.root,args.source]]
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.mode=='prepare' else '2,3',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1', PYTHONPATH=str(args.repo),
        TMPDIR=str(args.repo.parent/'tmp'), XDG_CACHE_HOME=str(args.repo.parent/'.cache'),
        HF_HOME=str(args.repo.parent/'.cache/huggingface'))
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'launch':
        for gpu in [2, 3]:
            gpu_idle(gpu)
        command=[sys.executable,str(Path(__file__).resolve()),'run','--repo',str(args.repo),
                 '--root',str(args.root),'--source',str(args.source),'--port',str(args.port)]
        with (args.root/'driver.log').open('a') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(driver_pid=child.pid, root=str(args.root))))
    else:
        try:
            run(args)
        except Exception as exc:
            previous=json.loads((args.root/'status.json').read_text())
            previous.update(stage='failed',updated_at=now(),error=str(exc))
            write_json(args.root/'status.json',previous)
            raise


if __name__ == '__main__':
    main()
