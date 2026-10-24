"""Record the user-authorized one-to-two GPU amendment after a clean stop."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

from voc_ab import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    from mmengine.config import Config
    root = args.root.resolve()
    archive = root/'gpu13_migration'
    stop = json.loads((archive/'stop_record.json').read_text())
    if not stop['old_driver_stopped']:
        raise RuntimeError('The old driver must stop at a complete checkpoint first')
    protocol_path = root/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['gpu_ids'] != [3] or (archive/'protocol.before.json').exists():
        raise RuntimeError('Expected an unamended single-GPU protocol')
    shutil.copy2(protocol_path, archive/'protocol.before.json')
    shutil.copy2(root/'setup_check.json', archive/'setup_check.before.json')
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    amendment = dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        authorization='User explicitly requested GPUs 1 and 3 on 2026-10-25',
        original_source_commit=protocol['source_commit'], source_commit=current_commit,
        old_gpu_ids=[3], new_gpu_ids=[1, 3], old_accumulation=2, new_accumulation=1,
        per_gpu_batch=2, effective_batch_before=4, effective_batch_after=4,
        resume_checkpoint=stop['retained_checkpoint'],
        checkpoint_sha256=sha256(stop['retained_checkpoint']),
        completed_epoch=stop['epoch'], historical_iteration=stop['iteration'],
        note='Topology, data shards and augmentation streams change; this is not bitwise continuation. Both A/B branches use identical two-GPU settings.')
    for plan in protocol['plans']:
        directory = Path(plan['directory'])
        if (directory/'stage_complete.json').exists():
            continue
        config = Path(plan['config'])
        shutil.copy2(config, archive/f"{plan['name']}.single_gpu.py")
        cfg = Config.fromfile(str(config))
        assert cfg.train_dataloader.batch_size == 2
        cfg.optim_wrapper.accumulative_counts = 1
        cfg.train_dataloader.num_workers = 2
        cfg.launcher = 'pytorch'
        cfg.model_wrapper_cfg = dict(type='MMDistributedDataParallel',
            broadcast_buffers=False, find_unused_parameters=True)
        cfg.diagnostic_protocol.update(gpu_ids=[1, 3], accumulation=1,
            effective_batch=4, topology_amended=True)
        cfg.dump(str(config))
        plan.update(config_sha256=sha256(config),
            microbatches_per_epoch=math.ceil(math.ceil(plan['train_images']/2)/2))
        if str(directory) == stop['original_status']['phase_dir']:
            plan.update(migration_resume_checkpoint=stop['retained_checkpoint'],
                        migration_resume_epoch=stop['epoch'])
    protocol.update(initial_source_commit=protocol['source_commit'], source_commit=current_commit,
                    gpu_ids=[1, 3], accumulation=1, gpu_migration=amendment)
    protocol['limitations'].append(amendment['note'])
    write_json(archive/'amendment.json', amendment)
    write_json(protocol_path, protocol)
    print(json.dumps(amendment, indent=2))


if __name__ == '__main__':
    main()
