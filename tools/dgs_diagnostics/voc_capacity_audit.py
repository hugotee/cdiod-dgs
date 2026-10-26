"""Function-preserving rank expansion and bounded capacity/KD observations."""
import copy
import hashlib
import math
from pathlib import Path

import torch
from mmdet.registry import HOOKS
from tools.dgs_diagnostics.voc_kd_audit import VOCFeatureKDAuditHook, write_json


def expand_state(state, seed=314159):
    """Copy old rank16 blocks; append random A and zero B, sharing init by layer."""
    generator = torch.Generator().manual_seed(seed)
    result, extra_a = {}, {}
    for key in sorted(state):
        value = state[key]
        if 'lora_A.' in key:
            assert value.ndim == 2 and value.shape[0] == 16, key
            stem = key.rsplit('.', 1)[0]
            if stem not in extra_a:
                tail = torch.empty_like(value)
                # Same Kaiming bound as nn.Linear(a=sqrt(5)); support the
                # server's older nn.init API without touching global RNG state.
                bound = 1. / math.sqrt(value.shape[1])
                tail.uniform_(-bound, bound, generator=generator)
                extra_a[stem] = tail
            result[key] = torch.cat([value, extra_a[stem]], dim=0)
        elif 'lora_B.' in key:
            assert value.ndim == 2 and value.shape[1] == 16, key
            result[key] = torch.cat([value, torch.zeros_like(value)], dim=1)
        else:
            result[key] = value
    return result


def verify_expansion(original, expanded):
    assert original.keys() == expanded.keys()
    maximum_error = 0.
    pairs = 0
    for key, value in original.items():
        if 'lora_A.' in key:
            b_key = key.replace('lora_A.', 'lora_B.')
            a, b = expanded[key], expanded[b_key]
            assert torch.equal(a[:16], value)
            assert torch.equal(b[:, :16], original[b_key])
            assert torch.count_nonzero(a[16:]) > 0
            assert torch.count_nonzero(b[:, 16:]) == 0
            old_weight = original[b_key].float() @ value.float() * 2.
            new_weight = b.float() @ a.float() * 2.
            error = float((old_weight-new_weight).abs().max())
            maximum_error = max(maximum_error, error)
            assert torch.allclose(old_weight, new_weight, atol=1e-6, rtol=1e-5), key
            pairs += 1
        elif 'lora_B.' not in key:
            assert torch.equal(value, expanded[key]), key
    assert pairs == 48
    return dict(passed=True, lora_pairs=pairs, old_rank=16, new_rank=32,
                old_alpha=32., new_alpha=64., scaling=2.,
                non_lora_tensors_identical=True, random_extra_A=True, zero_extra_B=True,
                maximum_effective_weight_absolute_error=maximum_error)


@HOOKS.register_module()
class VOCCapacityKDAuditHook(VOCFeatureKDAuditHook):
    def __init__(self, original_base, **kwargs):
        super().__init__(**kwargs)
        self.original_base = original_base

    def _write_capacity(self):
        write_json(self.directory/f'capacity_audit_rank{self.rank}.json', self.capacity)

    def before_train(self, runner):
        super().before_train(runner)
        self.audit.pop('same_initial_lora_as_original_A', None)
        self.audit['same_initial_lora_as_expanded_reference'] = True
        self.audit['reference_kind'] = 'function-preserving rank32 expansion of original A base'
        self._write()
        old = torch.load(self.original_base, map_location='cpu')['state_dict']
        actual = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        check = verify_expansion(old, actual)
        assert runner.cfg.model.moe_cfg.r == 32
        assert runner.cfg.model.moe_cfg.alpha == 64.
        expected = torch.load(self.reference_base, map_location='cpu')['state_dict']
        assert all(torch.equal(actual[k], v) for k, v in expected.items())
        del old, actual, expected
        self.capacity = dict(rank=self.rank, preflight=self.preflight, initial=check,
            trainable_lora_parameters=sum(p.numel() for _, p in self.student),
            teacher_lora_parameters=sum(p.numel() for _, p in self.teacher),
            final_lora_budget=1376256, B_final_lora_budget=1376256,
            exact_total_or_trainable_parameter_match=False,
            finite_nonzero_extra_B_gradient_observations=0,
            full_detector_prediction_check=None, passed=False)
        assert self.capacity['trainable_lora_parameters'] == 1376256
        self.gradient_handles = []
        for name, parameter in self.student:
            if 'lora_B.' in name:
                def observe(gradient):
                    tail = gradient[:, 16:]
                    if bool(torch.isfinite(tail).all()) and bool(tail.abs().sum() > 0):
                        self.capacity['finite_nonzero_extra_B_gradient_observations'] += 1
                self.gradient_handles.append(parameter.register_hook(observe))
        self._write_capacity()

    def _prediction_check(self, data_batch):
        """Compare actual detector predictions with rank16 slices, before updates.

        Only forward outputs are temporarily replaced; parameters, DDP buckets,
        optimizer state and random state are retained. Full precision avoids AMP
        rank-dependent rounding obscuring the initialization check.
        """
        from projects.DGS.dgs.layers.modules.group_lora import LoraLinearRep
        import random
        import numpy as np
        python_state, numpy_state = random.getstate(), np.random.get_state()
        was_training = self.model.training
        handles = []
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self.model.eval()
                with torch.no_grad(), torch.autocast('cuda', enabled=False):
                    expanded = self.model.test_step(copy.deepcopy(data_batch))
                    def old_forward(module, inputs, output):
                        task = module.task_ids[-1] if len(module.lora_B) > 1 else module.task_ids[0]
                        a = module.lora_A[str(task)][:16]
                        b = module.lora_B[str(task)][:, :16]
                        x = inputs[0]
                        return ((module.lora_dropout(x) @ a.T @ b.T)*module.scaling).to(x.dtype)
                    for module in self.model.modules():
                        if isinstance(module, LoraLinearRep):
                            handles.append(module.register_forward_hook(old_forward))
                    reference = self.model.test_step(copy.deepcopy(data_batch))
                assert len(expanded) == len(reference)
                max_score, max_box = 0., 0.
                for new, old in zip(expanded, reference):
                    x, y = new.pred_instances, old.pred_instances
                    assert torch.equal(x.labels, y.labels)
                    assert x.scores.shape == y.scores.shape
                    max_score = max(max_score, float((x.scores-y.scores).abs().max()))
                    max_box = max(max_box, float((x.bboxes-y.bboxes).abs().max()))
                    assert torch.allclose(x.scores, y.scores, atol=1e-5, rtol=1e-4)
                    assert torch.allclose(x.bboxes, y.bboxes, atol=.02, rtol=1e-4)
                return dict(passed=True, images=len(expanded), identical_labels=True,
                    maximum_score_absolute_error=max_score, maximum_box_absolute_error=max_box,
                    mode='full-detector FP32 prediction, rank32 versus original rank16 slices')
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if self.preflight and self.step == 0:
            self.capacity['full_detector_prediction_check'] = self._prediction_check(data_batch)
            self._write_capacity()
        super().before_train_iter(runner, batch_idx, data_batch)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        if self.step == self.audit_steps:
            b_norms = [float(p.detach()[:, 16:].float().norm()) for name, p in self.student
                       if 'lora_B.' in name]
            self.capacity['extra_B_norms_after_audit'] = b_norms
            self.capacity['extra_B_learned'] = any(v > 0 for v in b_norms)
            self.capacity['passed'] = bool(self.audit['passed'] and
                self.capacity['extra_B_learned'] and
                self.capacity['finite_nonzero_extra_B_gradient_observations'] > 0 and
                (not self.preflight or self.capacity['full_detector_prediction_check']['passed']))
            self._write_capacity()
            assert self.capacity['passed'], 'Expanded rank must actually learn'
            for handle in self.gradient_handles:
                handle.remove()


def cpu_check(path):
    """Exercise real LoRA class expansion, gradient access and group merging."""
    from projects.DGS.dgs.layers.modules.group_lora import LoraLinearRep
    torch.set_num_threads(2)
    torch.manual_seed(731)
    old = LoraLinearRep(7, 9, r=16, lora_alpha=32,
        task_ids=[0, 1, 2, 3], cur_task_id=3, group_cfg=dict(merge_method='ema'))
    with torch.no_grad():
        old.lora_B['0'].normal_(std=.02)
        old.lora_A['3'].copy_(old.lora_A['0'])
        old.lora_B['3'].copy_(old.lora_B['0'])
    new = LoraLinearRep(7, 9, r=32, lora_alpha=64,
        task_ids=[0, 1, 2, 3], cur_task_id=3, group_cfg=dict(merge_method='ema'))
    new.load_state_dict(expand_state(old.state_dict()))
    new.__freeze_params__(3)
    x = torch.randn(3, 7)
    assert torch.allclose(old(x), new(x), atol=1e-6, rtol=1e-5)
    new(x).square().sum().backward()
    norm = float(new.lora_B['3'].grad[:, 16:].norm())
    assert norm > 0 and math.isfinite(norm)
    assert new.lora_A['0'].grad is None and new.lora_B['0'].grad is None
    optimizer = torch.optim.SGD([p for p in new.parameters() if p.requires_grad], lr=.01)
    optimizer.step()
    assert new.lora_B['3'][:, 16:].abs().sum() > 0
    new.__rep__(delete=True)
    assert list(new.lora_A.keys()) == ['0']
    assert new.lora_A['0'].shape[0] == 32
    result = dict(passed=True, real_LoraLinearRep=True, function_preserved=True,
        extra_B_nonzero_gradient=norm, extra_B_updated=True, teacher_frozen=True,
        existing_merge_supports_rank32=True)
    write_json(path, result)
    return result
