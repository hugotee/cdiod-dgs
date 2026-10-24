# DTG group-diameter diagnostic preparation

The proposed constraint is a maximum **pairwise symmetric KL within the group
AFTER admitting the new task**. It is NOT max-minus-min of new-task distances.

For nearest old task j and candidate group G:

1. Preserve the original nearest-neighbor condition: d(new,j) < expand_th.
2. If max_group_kl is enabled, also require
   max(old group diameter, max(d(new,k) for k in G)) <= max_group_kl.
3. Rejecting the candidate creates a new group. Do not silently search other
   groups, change the KL estimator, change the OOD threshold, or change KD.

## Git and baseline

- Archived baseline: tag `dgs-repro-20261024`, commit `7658d10`.
- Observation-only instrumentation: commit `fa7aea5` (original decision intact).
- Optional constraint and CPU tools: commit `314e288`, followed by additional
  preflight checks and documentation on this branch.
- Working branch: `codex/dtg-diameter-diagnostics`.
- Unrelated pre-existing `work/` files are not part of this work.

## What has been checked

`five_phase/` contains offline simulations on the stored training-statistic
matrix. Six dataset-order permutations x caps [150, 200, 300] = 18 modified
simulations. None changes the original grouping. These are NOT 18 trainings or
AP measurements and do not show that the hypothesis is true or false generally.

The checks in `tests/test_dgs_grouping_diagnostics.py` cover the singleton case,
nearest-neighbor chaining, old-old diameter, threshold boundaries, off-switch
behavior, real five-task replay, and a synthetic intervention. The synthetic
case verifies code behavior, not a failure on the real benchmark.

```bash
python -m unittest discover -s tests -p test_dgs_grouping_diagnostics.py -v
python tools/dgs_diagnostics/simulate_grouping.py \
  --input experiments/cdiod/repro_20261024/saved_statistics_summary.json \
  --output-dir work_dirs/dtg_diagnostics/five_phase \
  --caps 150 200 300 --dataset-orders all
```

## Ten-task statistics: required before deciding on a training intervention

The server holds the ten-task annotations and the five-task per-image feature
caches. `prepare_cached_statistics.py` reuses embeddings by domain + image
filename, validates coverage and repeated-image consistency, then reconstructs
the last-level Gaussian statistics with the existing covariance/SVD operations.
It uses TRAIN annotations only and writes into a NEW diagnostic directory.
It never extracts GPU features, constructs a detector, or launches training.
Missing or inconsistent caches are reported as errors, not silently skipped.
Original per-file extraction manifests were not saved; the reuse assumption is
explicitly recorded and must be reviewed alongside the extraction-source hashes.

Run on the server ONLY after connectivity and the checked-out commit are verified:

```bash
python tools/dgs_diagnostics/prepare_cached_statistics.py \
  --repo "$PWD" --output-dir work_dirs/dtg_diagnostics/ten_phase/stats
python tools/dgs_diagnostics/simulate_grouping.py \
  --input work_dirs/dtg_diagnostics/ten_phase/stats/distance_matrix.json \
  --output-dir work_dirs/dtg_diagnostics/ten_phase/simulation \
  --caps 150 200 300 --dataset-orders all
```

These steps remain pending while the school SSH endpoint is unreachable. The
CPU cached-statistics script has received syntax review, but has NOT yet been
validated against the server's torch environment. Runtime predictor integration
also needs a CPU check on saved statistics after connectivity returns. Do not
report server deployment or ten-task results as completed until they succeed.

## Online record and constraint switches

In `model.domain_predictor_cfg`:

```python
grouping_trace_path = 'work_dirs/my_diagnostic/grouping_trace.jsonl'
max_group_kl = None   # observation-only; original decisions
# max_group_kl = 150  # enable the diameter constraint
```

Record-only and constrained runs must use separate NEW mapping paths. Loading
a completed task mapping intentionally skips expansion; it does not replay
decisions or retroactively repair existing groups. The isolated config
`projects/DGS/configs/CDIOD/dgs_diameter_diagnostic.py` is for predictor/mapping
checks, not a full training launch config. Set its task names, statistic paths
and mapping path before using adaptive_task_mapping.py.

## Training gate and paired-run specification

No training has been started by this preparation. Require a real task/order/cap
where admission changes before paying for a training comparison. Do not select
caps using validation/test AP. If there is no real trigger, report that result
and revisit the hypothesis rather than forcing a split with an arbitrary cap.

For an informative pair, freeze the common preceding checkpoint and record:
source commit, task order, data hashes, seed, global batch, epochs, LR policy,
prompt/evaluation class space, permitted historical state, and mapping snapshots.
Use per-GPU batch 2 / global batch 4 as a conservative initial candidate on
GPUs 2 and 3; its memory/runtime still needs checking for the chosen ten-task
stage. Keep the original dynamic new-group/consolidation training policies in
both methods; when their branch differs, this is a comparison of overall
grouping strategies, not a clean causal estimate of parameter sharing alone.
Report parameter/storage growth and training time along with old/new task AP.

Normal routing is the deployment metric. Forced-group routing is a diagnostic
reference only and must preserve image, text and evaluation rules. If a future
paired run trains an existing group, save base, end-of-training pre-merge new,
and merged states with a separately reviewed hook; do not label epoch 11 as
the exact missing end-of-training state. Original baseline artifacts remain
unchanged. Full-sequence comparisons, seeds and orders follow only after a
meaningful local intervention has been established.
