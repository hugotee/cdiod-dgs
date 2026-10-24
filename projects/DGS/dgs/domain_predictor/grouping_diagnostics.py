"""Pure-Python DTG decision records, shared by online and offline diagnostics."""
import itertools
import json
import math
from pathlib import Path


def candidate_group_record(task_id, mapping, distances, pair_distance, expand_th):
    """Describe adding a task to its nearest old task's group.

    ``distances[j]`` must be the SAME symmetric KL used by DTG. The previous
    group diameter includes old-old pairs, not only new-old pairs. A singleton
    has diameter zero by convention, but its new-old KL need not be zero.
    """
    if task_id in mapping:
        raise ValueError('A recorded task must not already have a group')
    if set(mapping) != set(range(task_id)) or len(distances) != task_id:
        raise ValueError('Expected contiguous prior task IDs and one distance each')
    values = [float(v) for v in distances]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Grouping needs finite prior-task distances')
    nearest = min(range(task_id), key=values.__getitem__)
    group = mapping[nearest]
    members = sorted(t for t, g in mapping.items() if g == group)
    old_pairs = [float(pair_distance(a, b))
                 for a, b in itertools.combinations(members, 2)]
    if not all(math.isfinite(v) for v in old_pairs):
        raise ValueError('Group pair distances must be finite')
    candidate_distances = [values[t] for t in members]
    old_diameter = max(old_pairs, default=0.0)
    new_max = max(candidate_distances)
    return dict(task_id=task_id, mapping_before=dict(mapping),
                nearest_task=nearest, candidate_group=group, members_before=members,
                group_size_before=len(members), all_prior_distances=values,
                candidate_distances={str(t): values[t] for t in members},
                min_kl=min(candidate_distances), max_kl=new_max,
                kl_range=new_max-min(candidate_distances),
                group_diameter_before=old_diameter,
                group_diameter_after=max(old_diameter, new_max),
                expand_th=float(expand_th),
                original_accepts=values[nearest] < expand_th)


def append_grouping_record(path, record):
    """The caller must restrict writes to rank zero. No training state changes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')


def apply_diameter_limit(record, max_group_kl=None):
    """Annotate a decision without mutating the observation or its mapping.

    None disables the constraint. Keep the nearest-neighbor test STRICT (<),
    and apply the diameter cap inclusively (<=). A rejected candidate starts
    a new group; this does not search for an alternative group.
    """
    if max_group_kl is not None:
        max_group_kl = float(max_group_kl)
        if not math.isfinite(max_group_kl) or max_group_kl <= 0:
            raise ValueError('max_group_kl must be finite and positive, or None')
    result = dict(record)
    result['max_group_kl'] = max_group_kl
    passes = max_group_kl is None or record['group_diameter_after'] <= max_group_kl
    result['diameter_accepts'] = passes
    result['accepts'] = record['original_accepts'] and passes
    result['diameter_rejected_original_merge'] = record['original_accepts'] and not passes
    result['decided_group'] = (record['candidate_group'] if result['accepts'] else
                               max(record['mapping_before'].values())+1)
    result['reason'] = ('nearest_distance' if not record['original_accepts'] else
                        'diameter_limit' if not passes else 'accepted')
    return result
