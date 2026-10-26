"""Observe the existing feature-KD path without changing model equations.

The experiment opts into the implemented inter-class path in its own config.
Repository defaults and all model/loss implementations remain unchanged.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import types

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import set_random_seed
from mmdet.registry import HOOKS


def write_json(path, value):
    import json
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


@HOOKS.register_module()
class VOCFeatureKDAuditHook(Hook):
    priority = 'BELOW_NORMAL'

    def __init__(self, reference_base, reference_branch, preflight=False, audit_steps=32):
        self.reference_base = reference_base
        self.reference_branch = Path(reference_branch)
        self.preflight = preflight
        self.audit_steps = audit_steps

    def _write(self):
        write_json(self.directory/f'feature_kd_audit_rank{self.rank}.json', self.audit)

    def before_train(self, runner):
        self.runner = runner
        self.rank = runner.rank
        self.directory = Path(runner.work_dir)
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        self.model = model
        head = model.bbox_head
        assert head.distn_cfg.feat_distn.type == 'inter-class'
        assert head.distn_cfg.feat_distn.subtype == 'opt1'
        assert head.loss_imgfeat_kd.loss_weight == 3.0
        assert head.loss_textfeat_kd.loss_weight == 5.0
        assert runner.world_size == 2
        assert runner.cfg.train_dataloader.batch_size == 2
        assert runner.cfg.optim_wrapper.accumulative_counts == 1
        reference = torch.load(self.reference_base, map_location='cpu')['state_dict']
        actual = model.state_dict()
        keys = [key for key in reference if 'lora_' in key]
        assert len(keys) == 96
        assert all(torch.equal(actual[key].detach().cpu(), reference[key]) for key in keys)
        del actual, reference
        self.student = [(name, parameter) for name, parameter in model.named_parameters()
                        if 'lora_' in name and name.endswith('.3')]
        self.teacher = [(name, parameter) for name, parameter in model.named_parameters()
                        if 'lora_' in name and name.endswith('.0')]
        assert len(self.student) == len(self.teacher) == 48
        assert all(parameter.requires_grad for _, parameter in self.student)
        assert all(not parameter.requires_grad for _, parameter in self.teacher)
        self.initial_student = {name: parameter.detach().cpu().clone()
                                for name, parameter in self.student}
        self.initial_teacher = {name: parameter.detach().cpu().clone()
                                for name, parameter in self.teacher}
        self.audit = dict(rank=self.rank, world_size=2, preflight=self.preflight,
            feature_type='inter-class', subtype='opt1', image_weight=3.0, text_weight=5.0,
            same_initial_lora_as_original_A=True, lora_tensors_verified=len(keys),
            teacher_lora_frozen=True, student_lora_tensors=48,
            relation_calls=0, calls_without_eligible_relations=0,
            loss_observations={'inter_text_loss':0, 'inter_query_loss':0},
            positive_loss_observations={'inter_text_loss':0, 'inter_query_loss':0},
            backward_observations={'inter_text_loss':0, 'inter_query_loss':0},
            positive_kd_feature_gradient_steps=0, teacher_features_detached=True,
            audit_steps=self.audit_steps, rows=[], passed=False)
        self.step = 0
        self.probing = False
        self.batch_row = None
        self.digest = hashlib.sha256()
        self.sample_count = 0
        self.paths = []
        original = head.inter_feat_distn_single
        def observe(this, *args, **kwargs):
            outputs = original(*args, **kwargs)
            self.audit['relation_calls'] += 1
            if self.batch_row is None:
                return outputs
            row = self.batch_row
            detached = not args[1].requires_grad and not args[7].requires_grad
            self.audit['teacher_features_detached'] &= detached
            if not detached:
                raise RuntimeError('Teacher features must be detached')
            losses = dict(zip(['inter_text_loss', 'inter_query_loss'], outputs))
            active = []
            for key, value in losses.items():
                if value is None:
                    continue
                scalar = float(value.detach())
                if not math.isfinite(scalar) or not value.requires_grad:
                    raise RuntimeError(f'Invalid or detached feature KD: {key}={scalar}')
                row['losses'][key] = scalar
                self.audit['loss_observations'][key] += 1
                self.audit['positive_loss_observations'][key] += int(scalar > 0)
                active.append(value)
                def backward(gradient, name=key, current=row):
                    if self.probing:
                        return
                    if not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError('Nonfinite KD backward signal')
                    nonzero = bool(gradient.detach().abs().sum() > 0)
                    current['backward'][name] = nonzero
                    self.audit['backward_observations'][name] += int(nonzero)
                value.register_hook(backward)
            if not active:
                self.audit['calls_without_eligible_relations'] += 1
            if active and self.preflight:
                # Stop at intermediate feature tensors. This does not invoke
                # parameter/DDP reducer hooks or accumulate parameter gradients.
                features = [value for value in [args[0], args[6]] if value.requires_grad]
                self.probing = True
                try:
                    gradients = torch.autograd.grad(sum(active), features,
                        retain_graph=True, allow_unused=True)
                finally:
                    self.probing = False
                norms = [float(value.detach().float().norm()) if value is not None else 0.
                         for value in gradients]
                if not all(math.isfinite(value) for value in norms):
                    raise RuntimeError('Nonfinite feature-KD gradients')
                row['kd_feature_gradient_norms'] = norms
                self.audit['positive_kd_feature_gradient_steps'] += int(any(value > 0 for value in norms))
            return outputs
        head.inter_feat_distn_single = types.MethodType(observe, head)
        if self.preflight:
            # Match the production audit hook's reset after model construction.
            set_random_seed(runner.cfg.randomness.seed, deterministic=False)
        self._write()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.batch_row = None
        if self.step < self.audit_steps:
            self.batch_row = dict(iteration=runner.iter+1, losses={}, backward={})
            self.audit['rows'].append(self.batch_row)
        if self.preflight and self.sample_count < 32:
            for tensor, sample in zip(data_batch['inputs'], data_batch['data_samples']):
                if self.sample_count == 32:
                    break
                self.paths.append(sample.img_path)
                self.digest.update(sample.img_path.encode())
                self.digest.update(str(tuple(tensor.shape)).encode())
                self.digest.update(tensor.contiguous().numpy().tobytes())
                self.sample_count += 1

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.step += 1
        if self.step == self.audit_steps:
            if self.preflight:
                import json
                reference = json.loads((self.reference_branch/f'first32_input_fingerprint_rank{self.rank}.json').read_text())
                self.audit['first32_input_sha256'] = self.digest.hexdigest()
                self.audit['same_first32_inputs_as_original_A'] = self.digest.hexdigest() == reference['sha256']
                assert self.audit['same_first32_inputs_as_original_A']
                self.audit['student_lora_updated'] = any(not torch.equal(
                    parameter.detach().cpu(), self.initial_student[name]) for name, parameter in self.student)
                self.audit['teacher_lora_unchanged'] = all(torch.equal(
                    parameter.detach().cpu(), self.initial_teacher[name]) for name, parameter in self.teacher)
                assert self.audit['student_lora_updated'] and self.audit['teacher_lora_unchanged']
                assert self.audit['positive_kd_feature_gradient_steps'] > 0
            else:
                import json
                reference = json.loads((self.reference_branch/f'first32_input_fingerprint_rank{self.rank}.json').read_text())
                actual = json.loads((self.directory/f'first32_input_fingerprint_rank{self.rank}.json').read_text())
                self.audit['same_first32_inputs_as_original_A'] = actual['sha256'] == reference['sha256']
                assert self.audit['same_first32_inputs_as_original_A']
            self.audit['passed'] = all(self.audit['positive_loss_observations'][key] > 0 and
                self.audit['backward_observations'][key] > 0 for key in self.audit['loss_observations'])
            self._write()
            if not self.audit['passed']:
                raise RuntimeError('Both feature-KD terms must be nonzero and enter backward within the audit window')
        elif self.step < self.audit_steps and (self.step < 4 or self.step % 8 == 0):
            self._write()

    def after_train(self, runner):
        assert self.audit['passed']
        self.audit['completed_training_iterations'] = runner.iter
        self._write()


def cpu_check(output):
    """Exercise actual guard, topology implementation, total loss and gradients."""
    from mmengine.config import ConfigDict
    from mmengine.model import BaseModel
    from mmengine.structures import InstanceData
    from mmdet.models.losses import L2Loss
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.DGS.dgs
    from projects.DGS.dgs.heads.gdino_head_inc_dgs import GroundingDINOHead_inc_DGS
    torch.set_num_threads(2)
    torch.manual_seed(731)
    head = object.__new__(GroundingDINOHead_inc_DGS)
    torch.nn.Module.__init__(head)
    head.distn_cfg = ConfigDict(feat_distn=dict(type='inter-intra', subtype='opt1'),
        label_distn=dict(loss_ld=dict(type='None')),
        query_distn=dict(type='seperate_queryinit', num_aux_query=12))
    head.ori_token_positive_maps = {1:[0], 2:[1], 3:[2]}
    head.loss_imgfeat_kd = L2Loss(loss_weight=3., reduction='mean')
    head.loss_textfeat_kd = L2Loss(loss_weight=5., reduction='mean')
    teacher_query = torch.randn(1, 2, 12, 256)*.2
    teacher_text = torch.randn(2, 3, 256)*.2
    student_query = torch.nn.Parameter(teacher_query+torch.randn_like(teacher_query)*.1)
    student_text = torch.nn.Parameter(teacher_text+torch.randn_like(teacher_text)*.1)
    logits = torch.full((1, 2, 12, 256), -12.)
    for query in range(12):
        logits[:, :, query, query % 3] = 8.
    boxes = torch.zeros(1, 2, 12, 4)
    instances = [InstanceData(labels=torch.tensor([0,1,2])) for _ in range(2)]
    args = (logits, boxes, instances, [{},{}], student_query, student_text,
            logits.clone(), boxes.clone(), instances, instances, teacher_query, teacher_text)
    assert head.loss_by_feat_old(*args) == {}, 'Original configuration unexpectedly enabled feature KD'
    head.distn_cfg.feat_distn.type = 'inter-class'
    losses = head.loss_by_feat_old(*args)
    assert set(losses) == {'inter_text_loss','inter_query_loss'}
    total, _ = BaseModel.parse_losses(head, losses)
    assert torch.allclose(total, sum(losses.values()))
    total.backward()
    gradients = [float(value.grad.norm()) for value in [student_query, student_text]]
    assert all(math.isfinite(value) and value > 0 for value in gradients)
    assert not teacher_query.requires_grad and not teacher_text.requires_grad
    sparse = [InstanceData(labels=torch.tensor([0,1])) for _ in range(2)]
    sparse_args = list(args)
    sparse_args[8] = sparse
    assert head.loss_by_feat_old(*sparse_args) == {}, 'Sparse-class behavior was changed'
    result = dict(passed=True, cpu_only=True, actual_guard_off_before=True,
        actual_guard_on_after=True, both_losses_present=True,
        losses={key:float(value.detach()) for key,value in losses.items()},
        total_loss_includes_both=True, student_gradient_norms=gradients,
        teacher_detached=True, fewer_than_three_pseudo_classes_skips_relations=True,
        equations_and_loss_implementation_unchanged=True)
    write_json(output, result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check', type=Path)
    parser.add_argument('--preflight-config', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if args.cpu_check:
        cpu_check(args.cpu_check)
    elif args.preflight_config:
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        import projects.DGS.dgs
        # Executed as __main__, register the hook once in this process.
        cfg = Config.fromfile(str(args.preflight_config))
        cfg.launcher = 'pytorch'
        torch.set_num_threads(4)
        Runner.from_cfg(cfg).train()
    else:
        parser.error('Choose --cpu-check or --preflight-config')
