"""Grouping-only config. Use adaptive_task_mapping.py, NOT tools/train.py.

The offline/preflight code builds only a predictor. Each simulation must use a
fresh task mapping; a completed mapping intentionally skips all new decisions.
"""
_base_ = './dgs_stage1.py'
model = dict(domain_predictor_cfg=dict(
    max_group_kl=None,  # None: original DTG. Set 150 to enable the new constraint.
    grouping_trace_path='work_dirs/CDIOD/diameter_diagnostic/grouping_trace.jsonl',
    task_id_mapping_path='work_dirs/CDIOD/diameter_diagnostic/task_id_mapping.yaml',
))
