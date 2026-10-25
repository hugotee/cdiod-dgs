# VOC four-task diagnostic: original sharing vs diameter cap 75

Authorized by the user on 2026-10-24 following the proposed small AP diagnostic.
This is one seed of a single-domain experiment, not full ten-task CDIOD.

- Order: PascalVOC_Task4-1, 2, 3, 4 (five classes per task).
- Shared prefix: train tasks 1--3 once, 12 epochs each.
- A: task 4 joins the original group (nearest threshold 150).
- B: task 4 creates a new group (nearest 150, maximum group diameter 75).
- Both branches load the identical task-3 checkpoint; its SHA-256 is recorded.
- GPU 2 is occupied by another user. Use only GPU 3, microbatch 2, gradient
  accumulation 2, nominal effective batch 4. Keep this fixed for all stages.
- Seed 1498185232, AMP, existing augmentations, LoRA rank16/alpha32,
  12 epochs/task, epoch11 LR decay, no automatic batch/LR/epoch changes.
- Preserve the existing DGS training branch policies: a new group uses stage1
  (LR8e-4), consolidation uses stage2 (LR5e-4). Initialization, training text,
  pseudo labels, distillation and merging differ with the selected branch.
- Retain the known reproduction configuration mismatch: feature-distillation
  type `inter-intra`, while its implementation branch checks `inter-class`.
  Pseudo-label and old-query paths remain active. This limitation must accompany
  interpretation; this run does not silently repair or claim exact paper KD.
- Disable encoder/backbone checkpointing consistently; batch2 was previously
  viable in the local reproduction. Stop on OOM rather than silently changing
  one branch's training budget.

Evaluation uses the full VOC validation image set and identical 20-class text
for the common prefix and both final branches. Evaluate task3 and A/B task4;
skip task1/2 validation to save time. Report normal-router AP@[.50:.95] for old
15 classes, new5 classes and all20 classes. Average the equal-five-class task APs
directly, retaining zero values; do not use the evaluator's positive-only avg.
Save prediction JSONs, individual-class metrics, parameter counts and runtime.

An audit hook with priority ABOVE_NORMAL runs before the original NORMAL
MergeHook to save exact `new_premerge` weights. Save raw full states for base,
new_premerge and merged. Checkpoint transforms never run during these snapshots.
A CPU check invokes the actual MMEngine hook order and actual DGS MergeHook on
a tiny module, and compares 32 augmented task4 inputs between A and B. Runtime
records repeat the input fingerprint test and compare prefix checkpoint hashes.

```bash
python tools/dgs_diagnostics/voc_ab.py prepare --repo "$PWD" \
  --root /absolute/new/experiment/root --stats /absolute/cached/ten/task/stats --gpu 3
python tools/dgs_diagnostics/check_voc_ab_setup_cpu.py --repo "$PWD" \
  --root /absolute/new/experiment/root
python tools/dgs_diagnostics/voc_ab.py launch --repo "$PWD" \
  --root /absolute/new/experiment/root --stats /absolute/cached/ten/task/stats --gpu 3
```

The bounded driver runs three shared stages then A and B sequentially, and
stops on failure. No scheduled automation or full-sequence sweep is installed.
Source/input/config hashes and protocol are frozen before launch. Same-phase
restart skips `moe_group_init` so it cannot overwrite a trained student; no
automatic OOM fallback mixes batch sizes. Finished stages are retained.

This first intervention is equivalent to forcing task4 independent. It checks
whether this sharing decision is useful, not whether cap75 is generally optimal.
B has an extra group. A positive result needs capacity-matched controls, other
split points/orders and multiple seeds before supporting a method-level claim.

## User-authorized GPU migration, 2026-10-25

The user subsequently requested GPUs 1 and 3. Retain completed task1 and stop
task2 at a full epoch checkpoint, preserving optimizer, scheduler and AMP state.
`stop_voc_at_checkpoint.py` validates ownership/command before signaling only
the experiment driver and keeps a separate copy of the recovery checkpoint.
`migrate_voc_dual.py` archives the initial protocol/configs and records an
amendment: 2 GPUs x batch2 x accumulation1 = effective batch4. Epoch budgets and
LR policies are unchanged; both future A/B branches use this two-GPU setting.
The existing prefix contains a single-GPU history, so this is not bitwise
continuation of its data/augmentation stream.

The audit verifies LoRA tensors, optimizer steps, AMP scaler, LR scheduler and
epoch/iteration state on each rank. Historical iteration counters are retained;
remaining iteration counts are recomputed from the new per-rank dataloader.
Only rank0 saves snapshots/metrics; each rank writes its own input-pairing audit.
`check_voc_dual_cpu.py` checks A/B input pairing with two actual Gloo CPU ranks.
Use `voc_ab.py launch ... --gpus 1 3 --port 29583` only after these checks pass.

## Completed result

All five training stages completed on 2026-10-25 at 11:34 Asia/Shanghai using
training source `b897608`, including the recorded initial single-GPU prefix.
Normal-router AP points (A vs B): old15 63.83 vs65.06, new5 64.93 vs64.52,
all20 64.11 vs64.93. B-A all20 is +0.82 points for this single seed/split.
Both origin checkpoint and per-rank first32 input pairing audits passed.
See [Chinese result report](结果汇总.md) and `completed_20261025/` for compact
metrics, configs, input hashes, parameter breakdown and checkpoint audits.
Large weights, predictions and dataset annotations remain on the server.
No additional training was queued and GPUs1/3 have been released.
