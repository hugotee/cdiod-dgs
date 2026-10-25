"""Read-only, strictly loaded VOC routing/merge diagnostics. No train calls."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
from collections import Counter

import torch
from mmengine.config import Config
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.utils import register_all_modules


def write_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def lora_digest(state):
    digest = hashlib.sha256()
    for key in sorted(key for key in state if 'lora_' in key):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_router(model, forced_group, rows):
    predictor = model.domain_predictor
    if not hasattr(predictor, '_diagnostic_original_forward'):
        predictor._diagnostic_original_forward = predictor.forward
    representative = None
    if forced_group is not None:
        members = [task for task, group in predictor.task_id_mapping.items() if group == forced_group]
        if not members:
            raise ValueError('Requested group does not exist')
        # The detector maps returned task_id to group_id. Returning group_id
        # directly would select the WRONG group in a mapping such as 3->1.
        representative = min(members)
    def forward(this, features, task_id=None, input_sample=None):
        prediction = representative if representative is not None else int(
            this._diagnostic_original_forward(features, task_id=task_id, input_sample=input_sample))
        group = this.task_id_mapping.get(prediction, -1)
        if input_sample is not None:
            rows.extend(dict(image_id=int(sample.img_id), predicted_task=int(prediction),
                             selected_group=int(group)) for sample in input_sample)
        return int(prediction)
    predictor.forward = types.MethodType(forward, predictor)
    return representative


@HOOKS.register_module()
class VOCInferenceAuditHook(Hook):
    def after_load_checkpoint(self, runner, checkpoint):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        state = checkpoint['state_dict']
        # This hook is the only custom load hook. No group initialization,
        # parameter merge or checkpoint transforms are registered.
        model.load_state_dict(state, strict=True)
        self.expected_digest = lora_digest(state)
        self.expected_lora_keys = sorted(key for key in state if 'lora_' in key)

    def before_test(self, runner):
        self.started = time.monotonic()
        self.model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        self.rows = []
        protocol = runner.cfg.inference_diagnostic
        actual = self.model.state_dict()
        if lora_digest(actual) != self.expected_digest:
            raise RuntimeError('Loaded LoRA weights differ from the saved state')
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        representative = configure_router(self.model, protocol.forced_group, self.rows)
        selected = []
        for module in self.model.moe_modules:
            if hasattr(module, 'experts'):
                selected.append([list(expert.lora_B.keys()) for expert in module.experts])
        audit = dict(rank=runner.rank, world_size=runner.world_size, strict_load=True,
            checkpoint=runner.cfg.load_from, case=protocol.case,
            forced_group=protocol.forced_group, forced_representative_task=representative,
            lora_digest=self.expected_digest, lora_tensor_count=len(self.expected_lora_keys),
            expert_parameter_ids=selected, parameters_trainable=0,
            training_started=False, optimizer_constructed=False,
            batch_size=runner.cfg.test_dataloader.batch_size, fp16=False)
        write_json(Path(runner.work_dir)/f'load_audit_rank{runner.rank}.json', audit)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if is_main_process() and (batch_idx < 3 or (batch_idx+1) % 50 == 0):
            count = batch_idx+1
            elapsed = time.monotonic()-self.started
            write_json(Path(runner.work_dir)/'progress.json', dict(stage='inference',
                local_batches=count, local_total=len(runner.test_dataloader),
                images_processed_approx=min(count*runner.world_size, len(runner.test_dataloader.dataset)),
                validation_images=len(runner.test_dataloader.dataset), elapsed_seconds=elapsed,
                seconds_per_image_per_rank=elapsed/count))

    def after_test_epoch(self, runner, metrics=None):
        if lora_digest(self.model.state_dict()) != self.expected_digest:
            raise RuntimeError('LoRA weights changed during diagnostic evaluation')
        counts = Counter(row['selected_group'] for row in self.rows)
        write_json(Path(runner.work_dir)/f'routes_rank{runner.rank}.json', dict(
            counts=dict(counts), rows=self.rows, lora_unchanged=True))
        if is_main_process():
            write_json(Path(runner.work_dir)/'metrics.json', metrics)
            write_json(Path(runner.work_dir)/'progress.json', dict(stage='complete',
                validation_images=len(runner.test_dataloader.dataset),
                elapsed_seconds=time.monotonic()-self.started, training_started=False))


def main():
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    torch.set_num_threads(4)
    cfg = Config.fromfile(sys.argv[1])
    if cfg.train_cfg is not None or cfg.optim_wrapper is not None:
        raise RuntimeError('This entry point permits evaluation only')
    cfg.launcher = 'pytorch'
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
