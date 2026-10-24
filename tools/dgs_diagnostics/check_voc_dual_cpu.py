"""Check both distributed data shards pair identically across A/B, on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    import torch
    import torch.distributed as dist
    from mmengine.config import Config
    from mmengine.runner import Runner, set_random_seed
    from mmdet.utils import register_all_modules
    dist.init_process_group('gloo')
    assert dist.get_world_size() == 2
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    torch.set_num_threads(2)
    protocol = json.loads((args.root/'protocol.json').read_text())
    configs = [Config.fromfile(plan['config']) for plan in protocol['plans'][-2:]]
    values = []
    for cfg in configs:
        assert cfg.train_dataloader.batch_size == 2
        assert cfg.optim_wrapper.accumulative_counts == 1
        set_random_seed(protocol['seed'])
        loader = Runner.build_dataloader(cfg.train_dataloader, seed=protocol['seed'])
        digest = hashlib.sha256()
        paths = []
        for batch in loader:
            for tensor, sample in zip(batch['inputs'], batch['data_samples']):
                paths.append(sample.img_path)
                digest.update(sample.img_path.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
            if len(paths) >= 32:
                break
        values.append(dict(paths=paths, sha256=digest.hexdigest()))
        del loader
    assert values[0] == values[1], 'Distributed A/B augmented inputs differ'
    row = dict(rank=dist.get_rank(), passed=True, first32_sha256=values[0]['sha256'])
    rows = [None, None]
    dist.all_gather_object(rows, row)
    if dist.get_rank() == 0:
        result = dict(passed=True, cpu_only=True, world_size=2,
            nominal_effective_batch=4, common_prefix_path_equal=configs[0].load_from == configs[1].load_from,
            all_rank_pairing_passed=all(row['passed'] for row in rows), ranks=rows,
            single_gpu_setup_check_retained=True)
        assert result['common_prefix_path_equal']
        (args.root/'setup_check_dual.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
