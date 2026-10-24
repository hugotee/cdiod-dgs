# Completed five-phase DGS workflow run

Completed 2026-10-24 02:56:58 Asia/Shanghai. Detector source commit:
`f7932720274d88b97c6f047806177911dedb1774`.

Order: DIOR-1, DIOR-2, VOC-1, VOC-2, RUOD. Seed: 1498185232.
Final COCO AP50:95: DIOR 64.5, PascalVOC 67.9, RUOD 62.6;
equal-domain mean 65.0. See `phase_results.json` for all five evaluations.

This is a completed workflow run with adjusted settings, not a strict replication
of all paper settings. Global batch changed from 2 to 8 after phase 2 epoch 11,
then to 4 after an OOM in phase 4. Activation recomputation was disabled at the
first acceleration switch. New-group LR is 8e-4, consolidation LR 5e-4; each task
uses 12 epochs. Exact archived configs are in `configs/`.

The baseline tag archives source, results, and configurations retrospectively.
It does not imply that this archival commit was used to launch the original run.
Runtime scripts are the on-disk snapshots after completion; the fast driver
received a small OOM-metadata persistence edit after it was launched, so the
snapshot is not a claim of exact in-memory driver identity. Training settings
must be read from the phase configs and recorded history above.

Known upstream mismatch retained: stage-2 `feat_distn.type='inter-intra'`, while
the explicit feature-distillation branch checks `'inter-class'`. Teacher-generated
pseudo labels remain active. Resolve this separately before strict method claims;
do not silently change it in the grouping ablation.

Final group membership is [0, 0, 1, 1, 2]. Per-task stored SVD Gaussian statistics
and checkpoint inventories are summarized in `saved_statistics_summary.json`.
Full weights, datasets, feature tensors, credentials, and large logs are not part
of this archive. Paths in archived configs describe the original environment.
The final checkpoint is `RUOD/epoch_12.pth` under the original experiment root.

End-of-training pre-merge adapters were not explicitly saved. Epoch-11 weights
are earlier training states and must not be presented as exact pre-merge states.
Baseline results remain immutable when later diagnostics are run.
