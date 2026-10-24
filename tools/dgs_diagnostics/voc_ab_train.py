"""Bounded VOC diagnostic training with exact pre/post-merge snapshots.

Only used by the explicit VOC A/B experiment launcher; existing training entry
points and default hooks are unchanged.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner, set_random_seed
from mmdet.registry import HOOKS
from mmdet.utils import register_all_modules


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def snapshot_model(model, path, metadata):
    """Raw full state: never run checkpoint transformations or merge here."""
    path = Path(path)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    temporary = path.with_suffix('.pth.tmp')
    torch.save(dict(state_dict=state, meta=metadata), temporary)
    temporary.replace(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                parameter_tensors=len(state),
                lora_keys=sorted(key for key in state if 'lora_' in key))


@HOOKS.register_module()
class VOCDiagnosticAuditHook(Hook):
    # MergeHook uses NORMAL (50). Capture end-of-training new before that hook.
    priority = 'ABOVE_NORMAL'

    def _model(self, runner):
        return runner.model.module if is_model_wrapper(runner.model) else runner.model

    def _snapshot(self, runner, label):
        model = self._model(runner)
        directory = Path(runner.work_dir)/'diagnostic_states'
        directory.mkdir(exist_ok=True)
        target = directory/f'{label}.pth'
        if label == 'base' and target.exists():
            return
        metadata = dict(label=label, epoch_index=runner.epoch,
                        completed_epochs=runner.epoch+(label == 'new_premerge'), iteration=runner.iter,
                        task_id=model.task_id, task_mapping=model.domain_predictor.task_id_mapping,
                        source_checkpoint=runner.cfg.load_from,
                        total_parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        info = snapshot_model(model, target, metadata)
        write_json(directory/f'{label}.json', dict(metadata, **info))

    def before_train(self, runner):
        if runner.world_size != 1:
            raise RuntimeError('This diagnostic protocol fixes one GPU and accumulation=2')
        model = self._model(runner)
        # Four-task VOC labels, rather than the built-in 5/10/13-task lookup.
        # This is used only for routing accuracy counters, never routing choices.
        model.domain_predictor.tasks = {i: i//5 for i in range(20)}
        self.started = time.monotonic()
        self.initial_iter = runner.iter
        self.fingerprint = hashlib.sha256()
        self.sample_count = 0
        self.paths = []
        audit = dict(world_size=runner.world_size, per_gpu_batch=runner.cfg.train_dataloader.batch_size,
                     accumulation=runner.cfg.optim_wrapper.accumulative_counts,
                     nominal_effective_batch=runner.cfg.train_dataloader.batch_size *
                                             runner.cfg.optim_wrapper.accumulative_counts,
                     microbatches_per_epoch=len(runner.train_dataloader),
                     max_epochs=runner.max_epochs, max_microbatches=runner.max_iters,
                     resume=bool(runner.cfg.resume), source_checkpoint=runner.cfg.load_from,
                     task_mapping=model.domain_predictor.task_id_mapping,
                     total_parameters=sum(p.numel() for p in model.parameters()),
                     trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        write_json(Path(runner.work_dir)/f'runtime_audit_iter{runner.iter}.json', audit)
        self._snapshot(runner, 'base')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Reset the data/dropout RNG after branch-dependent model construction.
        # Sampler and worker seeds are also fixed in the generated configs.
        if not runner.cfg.resume:
            set_random_seed(runner.cfg.randomness.seed, deterministic=False)
        runner.logger.info('VOC_DIAGNOSTIC_AUDIT %s', json.dumps(audit))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if self.sample_count >= 32:
            return
        for tensor, sample in zip(data_batch['inputs'], data_batch['data_samples']):
            if self.sample_count >= 32:
                break
            path = sample.img_path
            self.paths.append(path)
            self.fingerprint.update(path.encode())
            self.fingerprint.update(str(tuple(tensor.shape)).encode())
            self.fingerprint.update(tensor.contiguous().numpy().tobytes())
            self.sample_count += 1
        if self.sample_count == 32:
            write_json(Path(runner.work_dir)/'first32_input_fingerprint.json',
                       dict(images=self.paths, sha256=self.fingerprint.hexdigest(),
                            excludes_labels_because_branch_class_spaces_differ=True))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter < 4 or (runner.iter+1) % 50 == 0:
            loss = float(runner.message_hub.get_scalar('train/loss').current())
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            elapsed = time.monotonic()-self.started
            completed = runner.iter+1-self.initial_iter
            write_json(Path(runner.work_dir)/'progress.json', dict(
                stage='train', epoch=runner.epoch+1, iteration=runner.iter+1,
                max_iterations=runner.max_iters, max_epochs=runner.max_epochs,
                completed_microbatches=completed, elapsed_seconds=elapsed,
                seconds_per_microbatch=elapsed/completed, latest_loss=loss,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))

    def after_train_epoch(self, runner):
        if runner.epoch+1 == runner.max_epochs:
            self._snapshot(runner, 'new_premerge')

    def after_val_epoch(self, runner, metrics=None):
        write_json(Path(runner.work_dir)/'normal_route_metrics.json', metrics)

    def after_train(self, runner):
        self._snapshot(runner, 'merged')
        write_json(Path(runner.work_dir)/'training_complete.json', dict(
            completed=True, train_and_validation_seconds=time.monotonic()-self.started,
            final_epoch=runner.epoch, final_iteration=runner.iter,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


def main():
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    torch.set_num_threads(4)
    cfg = Config.fromfile(sys.argv[1])
    cfg.launcher = 'none'
    if cfg.resume:
        # On same-phase recovery, retain trained student weights and optimizer.
        for hook in cfg.custom_hooks:
            if hook.type == 'WeightsTransformHook':
                hook.cfg = [value for value in hook.cfg if value['type'] != 'moe_group_init']
    cfg.custom_hooks.append(dict(type='VOCDiagnosticAuditHook'))
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
