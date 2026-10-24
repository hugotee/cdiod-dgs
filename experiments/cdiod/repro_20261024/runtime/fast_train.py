"""Training entry point that audits recovery when the batch size changes."""
import json, math, os, sys, time
from pathlib import Path
os.chdir('/home/data/dq/cdiod-dgs')
sys.path.insert(0, os.getcwd())
import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
import projects.DGS.dgs

@HOOKS.register_module()
class FastRecoveryAuditHook(Hook):
    def after_load_checkpoint(self, runner, checkpoint):
        if not runner.cfg.resume:
            return
        # Preserve historical update numbers. Only the future number of batches
        # changes; epoch-based LR milestones still use the restored epoch.
        meta = checkpoint['meta']
        runner.train_loop._max_iters = meta['iter'] + (runner.max_epochs-meta['epoch']) * len(runner.train_dataloader)

    def before_train(self, runner):
        loader = runner.train_dataloader
        batch = loader.batch_sampler.batch_size
        assert runner.world_size == 2
        assert batch == runner.cfg.train_dataloader.batch_size
        proof = dict(rank=runner.rank, world_size=runner.world_size, per_gpu_batch=batch,
            global_batch=batch*2, iterations_per_epoch=len(loader), start_epoch=runner.epoch,
            start_iter=runner.iter, planned_end_iter=runner.max_iters, resume=bool(runner.cfg.resume),
            encoder_num_cp=runner.cfg.model.encoder.num_cp,
            backbone_with_cp=runner.cfg.model.backbone.with_cp)
        if runner.cfg.resume:
            saved = torch.load(runner.cfg.load_from, map_location='cpu')
            model = runner.model.module if is_model_wrapper(runner.model) else runner.model
            actual = model.state_dict()
            keys = [k for k in saved['state_dict'] if 'lora_A' in k or 'lora_B' in k]
            assert keys and all(torch.equal(actual[k].cpu(), saved['state_dict'][k]) for k in keys)
            opt = runner.optim_wrapper.state_dict()
            assert opt['loss_scaler'] == saved['optimizer']['loss_scaler']
            assert [float(v['step']) for v in opt['state'].values() if 'step' in v] == [float(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
            assert [v.state_dict() for v in runner.param_schedulers] == saved['param_schedulers']
            assert runner.epoch == saved['meta']['epoch'] and runner.iter == saved['meta']['iter']
            proof.update(lora_tensors_verified=len(keys), optimizer_restored=True,
                scheduler_restored=True, amp_restored=True, checkpoint=runner.cfg.load_from)
            del actual, saved, opt
        self.start = time.monotonic()
        self.proof = proof
        torch.cuda.reset_peak_memory_stats()
        Path(runner.work_dir, f'fast_restore_rank{runner.rank}.json').write_text(json.dumps(proof, indent=2)+'\n')
        runner.logger.info('FAST_RECOVERY_AUDIT_OK %s', json.dumps(proof))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if batch_idx == 49:
            loss = float(runner.message_hub.get_scalar('train/loss').current())
            assert math.isfinite(loss)
            result = dict(self.proof, observed_iter=runner.iter+1, current_epoch=runner.epoch+1,
                fifty_steps_completed=True, elapsed_seconds=time.monotonic()-self.start,
                latest_loss=loss, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            Path(runner.work_dir, f'fast_running_rank{runner.rank}.json').write_text(json.dumps(result, indent=2)+'\n')

cfg = Config.fromfile(sys.argv[1])
cfg.launcher = 'pytorch'
cfg.optim_wrapper.type = 'AmpOptimWrapper'
cfg.optim_wrapper.loss_scale = 'dynamic'
cfg.custom_hooks.append(dict(type='FastRecoveryAuditHook'))
Runner.from_cfg(cfg).train()
