"""Prepare/launch one seed of a bounded four-task VOC A/B diagnostic.

Three shared prefix stages, followed by two alternative fourth stages. This is
not a ten-stage CDIOD reproduction. No hyperparameter search or auto OOM changes.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

TASKS = [f'PascalVOC_Task4-{i}' for i in range(1, 5)]
SEED = 1498185232
CHILD = None


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def plans(root):
    return [dict(name=f'prefix{i+1}', task_id=i, cap=None,
                 directory=str(root/'prefix'/TASKS[i])) for i in range(3)] + [
        dict(name='A_original', task_id=3, cap=None, directory=str(root/'branches/A_original')),
        dict(name='B_diameter75', task_id=3, cap=75, directory=str(root/'branches/B_diameter75'))]


def prepare(args):
    from mmengine.config import Config, ConfigDict
    from mmengine.logging import MMLogger
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules
    import torch
    import yaml
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    from projects.DGS.dgs.domain_predictor.svd_domain_predictor import AdaptiveDomainPredictor_SVD
    torch.set_num_threads(4)
    root, repo = args.root, args.repo
    if root.exists():
        raise FileExistsError('Use a NEW experiment root for preparation')
    root.mkdir(parents=True)
    MMLogger.get_instance('voc_ab_prepare')
    weight = repo/'weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
    if not weight.is_file() or not (repo/'weights/bert-base-uncased').is_dir():
        raise FileNotFoundError('Pretrained detector and local BERT must exist')
    if shutil.disk_usage(root).free < 25*2**30:
        raise RuntimeError('Require at least 25 GiB of free space')
    stats = root/'frozen_stats'
    stats.mkdir()
    hashes = {}
    for task in TASKS:
        source = args.stats/f'{task}.pt'
        shutil.copy2(source, stats/source.name)
        hashes[str(stats/source.name)] = sha256(stats/source.name)
    matrix = json.loads((args.stats/'distance_matrix.json').read_text())
    indexes = [matrix['tasks'].index(task) for task in TASKS]
    write_json(stats/'distance_matrix.json', dict(tasks=TASKS, distance_matrix=[
        [matrix['distance_matrix'][i][j] for j in indexes] for i in indexes],
        source=str(args.stats/'distance_matrix.json'), source_sha256=sha256(args.stats/'distance_matrix.json')))
    prepared = []
    for plan in plans(root):
        i = plan['task_id']
        directory = Path(plan['directory'])
        directory.mkdir(parents=True)
        mapping = directory/'task_id_mapping.yaml'
        mapping.write_text(yaml.safe_dump({j: 0 for j in range(max(1, i))}))
        trace = directory/'grouping_trace.jsonl'
        for task_id in range(i+1):
            predictor_cfg = ConfigDict(type='svd', multilevel=False, expand_th=150,
                ood_th=500, min_eig_ratio=1e-3, max_group_kl=plan['cap'],
                stats_path=str(stats), feat_path='UNUSED_CACHED_STATISTICS_REQUIRED',
                task_id_mapping_path=str(mapping), grouping_trace_path=str(trace))
            AdaptiveDomainPredictor_SVD(predictor_cfg, num_tasks=4, moe_modules=[],
                task_id=task_id, seen_tasks=TASKS[:task_id+1], work_dirs=str(directory))
        actual = yaml.safe_load(mapping.read_text())
        expected = {j: (1 if j == 3 and plan['cap'] == 75 else 0) for j in range(i+1)}
        if actual != expected:
            raise AssertionError((plan['name'], actual, expected))
        shared = sum(group == actual[i] for group in actual.values()) > 1
        cfg = Config.fromfile(str(repo/'projects/DGS/configs/CDIOD'/
                                  ('dgs_stage2.py' if shared else 'dgs_stage1.py')))
        cfg.work_dir = str(directory)
        cfg.model.update(num_tasks=4, task_id=i, seen_tasks=TASKS[:i+1])
        cfg.model.language_model.name = str(repo/'weights/bert-base-uncased')
        cfg.model.encoder.num_cp = 0
        cfg.model.backbone.with_cp = False
        cfg.model.domain_predictor_cfg.update(stats_path=str(stats),
            feat_path='UNUSED_CACHED_STATISTICS_REQUIRED', task_id_mapping_path=str(mapping),
            max_group_kl=plan['cap'], grouping_trace_path=str(trace))
        cfg.train_dataloader.update(batch_size=2, num_workers=4, persistent_workers=True)
        cfg.train_dataloader.dataset.update(data_root='data/CDIOD/PascalVOC/',
            ann_file=f'train/annotations_without_background_Task4-{i+1}.json',
            metainfo=','.join(TASKS[:i+1]) if shared else TASKS[i],
            seen_tasks=','.join(TASKS[:i+1]))
        if shared:
            cfg.train_dataloader.dataset.distn_cfg.task_id_mapping_path = str(mapping)
        # Identical 20-class prompt and full validation image set in all arms,
        # including the shared-prefix reference before task 4.
        validation = copy.deepcopy(cfg.val_dataloader.dataset.datasets[1])
        validation.metainfo = ('PascalVOC',)
        cfg.val_dataloader.dataset.datasets = [validation]
        cfg.val_dataloader.update(batch_size=1, num_workers=2, persistent_workers=True)
        metric = copy.deepcopy(cfg.val_evaluator.metrics[1])
        metric.classwise = True
        metric.outfile_prefix = str(directory/'validation_predictions')
        cfg.val_evaluator.update(metrics=[metric], dataset_prefixes=['PascalVOC'],
                                 evaluation_tasks=','.join(TASKS))
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
        cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
        if i < 2:
            cfg.val_dataloader = cfg.val_evaluator = cfg.val_cfg = None
        cfg.optim_wrapper.update(type='AmpOptimWrapper', loss_scale='dynamic', accumulative_counts=2)
        cfg.randomness = dict(seed=SEED, diff_rank_seed=False, deterministic=False)
        cfg.auto_scale_lr.enable = False
        cfg.default_hooks.checkpoint.update(save_optimizer=True, save_param_scheduler=True,
                                            max_keep_ckpts=2)
        cfg.default_hooks.logger.interval = 50
        cfg.load_from = str(weight if i == 0 else root/'prefix'/TASKS[i-1]/'epoch_12.pth')
        cfg.resume = False
        cfg.diagnostic_protocol = dict(kind='VOC_single_domain_AB', seed=SEED,
            effective_batch=4, microbatch=2, accumulation=2, gpu_ids=[args.gpu],
            fixed_data_and_augmentation_seed=True, full_validation_prompt_classes=20,
            primary_metric='normal routing AP', cap=plan['cap'],
            feature_distillation_configuration='inter-intra retained; implementation checks inter-class')
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
        n_images, n_classes = len(dataset), len(dataset.metainfo['classes'])
        del dataset
        ann = repo/cfg.train_dataloader.dataset.data_root/cfg.train_dataloader.dataset.ann_file
        hashes[str(ann)] = sha256(ann)
        config = directory/'train_config.py'
        cfg.dump(str(config))
        plan.update(mapping=actual, training_policy='consolidation' if shared else 'new_group',
                    train_images=n_images, training_prompt_classes=n_classes,
                    microbatches_per_epoch=math.ceil(n_images/2), epochs=12,
                    lr=cfg.optim_wrapper.optimizer.lr, config=str(config),
                    config_sha256=sha256(config))
        prepared.append(plan)
    annotation = repo/'data/CDIOD/PascalVOC/valid/annotations_without_background.json'
    hashes[str(annotation)] = sha256(annotation)
    hashes[str(weight)] = sha256(weight)
    protocol = dict(created_at=now(), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), tasks=TASKS, seed=SEED,
        gpu_ids=[args.gpu], effective_batch=4, microbatch=2, accumulation=2,
        input_sha256=hashes, plans=prepared, training_started=False,
        evaluation='full VOC validation, fixed 20-class text, normal router, AP@[.50:.95]',
        limitations=[
            'Single seed, single-domain diagnostic; not the full CDIOD benchmark.',
            'Cap75 is exploratory, selected using training-statistic grouping, not validation AP.',
            'A/B preserves DGS branch policies: LR, initialization, prompts, distillation and merging differ.',
            'B adds one group; AP gain alone does not establish better split selection at equal capacity.',
            'Known feature-distillation type mismatch is retained from the reproduction code.',
            'Nominal effective batch4; final accumulation window can have fewer images.',
            'Seed and first32 augmented input hashes are recorded; CUDA arithmetic is not bitwise deterministic.'])
    write_json(root/'protocol.json', protocol)
    write_json(root/'status.json', dict(stage='prepared', updated_at=now(), training_started=False))
    print(json.dumps(dict(root=str(root), plans=prepared), indent=2))


def collect(directory):
    directory = Path(directory)
    metrics = json.loads((directory/'normal_route_metrics.json').read_text())
    aps = [metrics[f'PascalVOC/{task}/coco/bbox_mAP'] for task in TASKS]
    if not all(math.isfinite(value) and value >= 0 for value in aps):
        raise RuntimeError('Invalid AP values; zero AP must never be silently dropped')
    return dict(old15_ap=sum(aps[:3])/3, new5_ap=aps[3], all20_ap=sum(aps)/4,
                per_task_ap=dict(zip(TASKS, aps)), raw_metrics=metrics,
                checkpoint=str(directory/'epoch_12.pth'), checkpoint_sha256=sha256(directory/'epoch_12.pth'),
                runtime=json.loads((directory/'training_complete.json').read_text()))


def gpu_idle(gpu):
    memory = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if memory > 512:
        raise RuntimeError(f'GPU{gpu} has {memory} MiB in use; do not collide with another job')


def run(args):
    global CHILD
    root = args.root
    lock = (root/'driver.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((root/'protocol.json').read_text())
    check_file = 'setup_check_dual.json' if len(args.gpus) == 2 else 'setup_check.json'
    if not json.loads((root/check_file).read_text())['passed']:
        raise RuntimeError('CPU setup checks must pass before training')
    if protocol['gpu_ids'] != args.gpus:
        raise RuntimeError('GPU assignment must match the prepared protocol')
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if current_commit != protocol['source_commit']:
        raise RuntimeError('Source commit differs from the prepared protocol')
    for path, expected_digest in protocol['input_sha256'].items():
        if sha256(path) != expected_digest:
            raise RuntimeError(f'Prepared input changed: {path}')
    def status(stage, **kwargs):
        write_json(root/'status.json', dict(stage=stage, updated_at=now(),
            driver_pid=os.getpid(), training_started=True, **kwargs))
    def terminate(signum, frame):
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
        status('interrupted', signal=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    (root/'driver.pid').write_text(str(os.getpid())+'\n')
    reference_hash = None
    for plan in protocol['plans']:
        directory = Path(plan['directory'])
        if (directory/'stage_complete.json').exists():
            continue
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != current_commit:
            raise RuntimeError('Source commit changed during the run; refusing to mix implementations')
        if sha256(plan['config']) != plan['config_sha256']:
            raise RuntimeError('Prepared config changed; do not mix protocols')
        from mmengine.config import Config
        cfg = Config.fromfile(plan['config'])
        if not Path(cfg.load_from).is_file():
            raise FileNotFoundError(cfg.load_from)
        source_hash = sha256(cfg.load_from)
        if plan['task_id'] == 3:
            if reference_hash is not None and source_hash != reference_hash:
                raise RuntimeError('A/B prefix checkpoint differs')
            reference_hash = source_hash
            write_json(directory/'branch_origin.json', dict(
                checkpoint=cfg.load_from, sha256=source_hash))
        for gpu in args.gpus:
            gpu_idle(gpu)
        if shutil.disk_usage(root).free < 8*2**30:
            raise RuntimeError('Insufficient disk space for snapshots and checkpoints')
        # Same-phase recovery is explicit and never changes batch or LR.
        checkpoints = list(directory.glob('epoch_*.pth'))
        if checkpoints:
            latest = max(checkpoints, key=lambda value: int(value.stem.split('_')[-1]))
            if latest.stem == 'epoch_12':
                raise RuntimeError('Final weights exist without a stage marker; inspect evaluation before recovery')
            cfg.resume = True
            cfg.load_from = str(latest)
            retained = plan.get('migration_resume_checkpoint')
            if retained and int(latest.stem.split('_')[-1]) == plan['migration_resume_epoch']:
                cfg.load_from = retained
            active_config = directory/'resume_config.py'
            cfg.dump(str(active_config))
        else:
            active_config = Path(plan['config'])
        entry = str(args.repo/'tools/dgs_diagnostics/voc_ab_train.py')
        command = [sys.executable, entry, str(active_config)]
        if len(args.gpus) == 2:
            command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
                       f'--master_port={args.port}', entry, str(active_config)]
        log = directory/('train.dual.log' if len(args.gpus) == 2 else 'train.log')
        status('train', name=plan['name'], task_id=plan['task_id'], phase_dir=str(directory),
               command=command, log=str(log), gpu_ids=args.gpus)
        with log.open('a') as stream:
            CHILD = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            status('train', name=plan['name'], task_id=plan['task_id'], phase_dir=str(directory),
                   child_pid=CHILD.pid, log=str(log), gpu_ids=args.gpus)
            code = CHILD.wait()
        CHILD = None
        if code:
            raise RuntimeError(f"{plan['name']} failed with code {code}; inspect {log}")
        if not (directory/'epoch_12.pth').is_file() or not (directory/'training_complete.json').is_file():
            raise RuntimeError('Missing final checkpoint or completion audit')
        result = collect(directory) if plan['task_id'] >= 2 else dict(
            checkpoint=str(directory/'epoch_12.pth'), evaluation_skipped_to_save_compute=True)
        write_json(directory/'stage_complete.json', result)
        status('stage_complete', name=plan['name'])
    a = json.loads((root/'branches/A_original/stage_complete.json').read_text())
    b = json.loads((root/'branches/B_diameter75/stage_complete.json').read_text())
    origins = [json.loads((root/'branches'/name/'branch_origin.json').read_text())['sha256']
               for name in ['A_original', 'B_diameter75']]
    fingerprints = [json.loads((root/'branches'/name/'first32_input_fingerprint.json').read_text())['sha256']
                    for name in ['A_original', 'B_diameter75']]
    rank_pairing = {}
    if len(args.gpus) == 2:
        for rank in range(2):
            values = [json.loads((root/'branches'/name/f'first32_input_fingerprint_rank{rank}.json').read_text())['sha256']
                      for name in ['A_original', 'B_diameter75']]
            rank_pairing[str(rank)] = values[0] == values[1]
    write_json(root/'comparison.json', dict(A=a, B=b, single_seed=True,
        origin_checkpoints_equal=origins[0] == origins[1],
        first32_augmented_inputs_equal=fingerprints[0] == fingerprints[1],
        rank_input_pairing=rank_pairing,
        delta_AP_points_B_minus_A={key:100*(b[key]-a[key]) for key in ['old15_ap', 'new5_ap', 'all20_ap']}))
    if origins[0] != origins[1] or fingerprints[0] != fingerprints[1] or not all(rank_pairing.values()):
        raise RuntimeError('Pairing audit failed; inspect comparison.json before interpreting AP')
    status('complete', comparison=str(root/'comparison.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'launch'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--port', type=int, default=29583)
    args = parser.parse_args()
    args.gpus = args.gpus or [args.gpu]
    if len(args.gpus) not in [1, 2] or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Use one GPU or two distinct GPUs')
    if args.mode == 'prepare' and len(args.gpus) != 1:
        raise ValueError('Use the recorded migration procedure to amend an existing protocol')
    args.repo, args.root, args.stats = (value.resolve() for value in [args.repo, args.root, args.stats])
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.mode == 'prepare' else ','.join(map(str, args.gpus)),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
        TMPDIR=str(args.repo.parent/'tmp'), XDG_CACHE_HOME=str(args.repo.parent/'.cache'),
        TORCH_HOME=str(args.repo.parent/'.cache/torch'), HF_HOME=str(args.repo.parent/'.cache/huggingface'))
    os.environ['PYTHONPATH'] = str(args.repo)
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'launch':
        for gpu in args.gpus:
            gpu_idle(gpu)
        command = [sys.executable, str(Path(__file__).resolve()), 'run', '--repo', str(args.repo),
                   '--root', str(args.root), '--stats', str(args.stats), '--gpus',
                   *map(str, args.gpus), '--port', str(args.port)]
        with (args.root/'driver.log').open('a') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(driver_pid=process.pid, root=str(args.root))))
    else:
        try:
            run(args)
        except Exception as exc:
            last = json.loads((args.root/'status.json').read_text())
            last.update(stage='failed', updated_at=now(), driver_pid=os.getpid(), error=str(exc))
            write_json(args.root/'status.json', last)
            raise


if __name__ == '__main__':
    main()
