# VOC routing and merge diagnostics (evaluation only)

The user authorized these diagnostics on GPUs 2 and 3 after the completed
single-seed VOC A/B experiment. No model training, optimizer, KD repair or
parameter updates are performed. All original weights/mappings remain intact.

Six full-VOC passes, with identical 3422 images, 20-class text, float32
evaluation and batch1 per GPU:

1. A merged, original router: reproduce the recorded A AP as a loading gate.
2. B final, original router: reproduce the recorded B AP as a loading gate.
3. A before task4 training, fixed group0.
4. A end-of-training premerge, fixed group0.
5. A merged, fixed group0.
6. B final, fixed old group0: report old15 AP against normal routing.

Compare 3->4 to locate training changes and 4->5 to locate merge changes, with
the route held fixed. The normal-vs-fixed A comparison also exposes OOD fallback
effects because the original A mapping contains just one group. Fixed routes use
task information and are diagnostic references, not deployment scores or
guaranteed upper bounds. Only the old15 metric is the target of B's fixed-old
pass; its other metrics are retained for audit, not advertised as oracle AP.

Merged A has only base LoRA0. For evaluation its group construction uses
`merge_method='none'`, which omits the absent temporary current expert.
Raw base/premerge A retains LoRA0 and3 with `ema` construction. All saved tensor
keys/shapes must load strictly; no group-init/weight-transform/merge hooks run.
The forward selection equations and saved weights are unchanged. Per-rank
load audits record LoRA hashes and expert IDs, and assert unchanged hashes
after evaluation. Forced group0 returns representative task0; forcing group1
would need task3 in B's mapping, not task1.

The CPU preflight checks actual checkpoint base cloning, preserved old tensors,
trained student tensors and the saved .2old+.8new parameter merge. It constructs
the three real model structures, strictly loads weights and tests task/group
selection. Full normal-route references must match old/new/all AP within0.05
points before the fixed-route runs proceed.

```bash
python tools/dgs_diagnostics/voc_inference.py prepare --repo "$PWD" \
  --source /absolute/completed/voc_ab --root /absolute/new/inference/root
python tools/dgs_diagnostics/check_voc_inference_cpu.py --repo "$PWD" \
  --root /absolute/new/inference/root
python tools/dgs_diagnostics/voc_inference.py launch --repo "$PWD" \
  --source /absolute/completed/voc_ab --root /absolute/new/inference/root
```

The bounded background driver checks free GPUs, source/config/checkpoint hashes,
load proofs, exact validation coverage and fixed-group consistency. It stops on
any mismatch. Results go into a new directory; large private prediction JSONs
are not committed. The known feature-KD type mismatch from training is recorded
as an interpretation limit; this inference experiment cannot repair past training.
