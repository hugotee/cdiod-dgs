"""CPU checks of real data, augmented input pairing and merge-snapshot order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    import torch
    from mmengine.config import Config
    from mmengine.logging import MMLogger
    from mmengine.runner import Runner, set_random_seed
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    from projects.DGS.dgs.hooks.merge_hook import MergeHook
    from voc_ab_train import VOCDiagnosticAuditHook
    torch.set_num_threads(4)
    MMLogger.get_instance('voc_ab_cpu_check')
    root = args.root
    protocol = json.loads((root/'protocol.json').read_text())
    configs = [Config.fromfile(plan['config']) for plan in protocol['plans']]
    val = DATASETS.build(configs[-1].val_dataloader.dataset)
    assert len(val.datasets[0].metainfo['classes']) == 20
    validation_images = len(val)
    del val
    fingerprints = []
    for cfg in configs[-2:]:
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
        fingerprints.append(dict(paths=paths, sha256=digest.hexdigest()))
        del loader
    assert fingerprints[0] == fingerprints[1], 'Paired augmented training inputs differ'
    assert configs[-2].load_from == configs[-1].load_from
    assert configs[-2].model.domain_predictor_cfg.max_group_kl is None
    assert configs[-1].model.domain_predictor_cfg.max_group_kl == 75

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_old = torch.nn.Parameter(torch.tensor([10.]))
            self.lora_new = torch.nn.Parameter(torch.tensor([2.]))
            self.task_id = 3
            self.domain_predictor = SimpleNamespace(task_id_mapping={0: 0, 1: 0, 2: 0, 3: 0})
        def __rep__(self, delete):
            self.lora_old.data.copy_(.2*self.lora_old.data + .8*self.lora_new.data)
            if delete:
                del self.lora_new

    # Exercise MMEngine's actual hook sorting/calling, followed by the real
    # DGS MergeHook on a tiny module, so the snapshot cannot silently run late.
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        runner = object.__new__(Runner)
        runner._hooks = []
        runner._work_dir = temporary
        runner._train_loop = SimpleNamespace(epoch=11, max_epochs=12, iter=1000, max_iters=1000)
        runner.model = Toy()
        runner.cfg = Config(dict(load_from='shared_prefix_test.pth'))
        audit = VOCDiagnosticAuditHook()
        audit.started = time.monotonic()
        runner.register_hook(MergeHook(dict(type='ema')))
        runner.register_hook(audit)
        runner.call_hook('after_train_epoch')
        runner._train_loop.epoch = 12
        runner.call_hook('after_train')
        before = torch.load(Path(temporary)/'diagnostic_states/new_premerge.pth', map_location='cpu')
        after = torch.load(Path(temporary)/'diagnostic_states/merged.pth', map_location='cpu')
        assert before['state_dict']['lora_new'].item() == 2
        assert before['state_dict']['lora_old'].item() == 10
        assert 'lora_new' not in after['state_dict']
        assert abs(after['state_dict']['lora_old'].item()-3.6) < 1e-5
        assert before['meta']['completed_epochs'] == after['meta']['completed_epochs'] == 12
    result = dict(passed=True, cpu_only=True, validation_images=validation_images,
                  fixed_prompt_classes=20, common_prefix_path_equal=True,
                  first32_augmented_inputs_equal=True, fingerprint=fingerprints[0]['sha256'],
                  actual_hook_order_verified=True, training_started=False)
    (root/'setup_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
