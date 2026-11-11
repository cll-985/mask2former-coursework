"""Pure mathematical simulations, not COCO data evaluation or new-method results.

Run: python offline_tools/mva_matching_sensitivity_math.py
No packages, datasets, model outputs, GPU, or network are used.
"""
from bisect import bisect_left
from itertools import combinations, permutations, product
import json
from random import Random
import sys


def greedy_match(rows, ground_truth_count):
    available = set(range(ground_truth_count))
    assignments, matched = [], []
    for preference in rows:
        chosen = next((g for g in preference if g in available), None)
        assignments.append(chosen)
        matched.append(chosen is not None)
        if chosen is not None:
            available.remove(chosen)
    return assignments, matched


def check_single_row(rows, edited_position, new_preference, ground_truth_count):
    before_ids, before = greedy_match(rows, ground_truth_count)
    modified = list(rows)
    modified[edited_position] = new_preference
    after_ids, after = greedy_match(modified, ground_truth_count)
    assert before[:edited_position] == after[:edited_position]
    assert before_ids[:edited_position] == after_ids[:edited_position]
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(changed) <= 2, (rows, edited_position, new_preference, before_ids, after_ids)
    assert abs(sum(after)-sum(before)) <= 1
    if len(changed) == 2:
        assert int(after[changed[0]])-int(before[changed[0]]) == -(int(after[changed[1]])-int(before[changed[1]]))
    if changed:
        first_sign = int(after[changed[0]])-int(before[changed[0]])
        prefix_difference = 0
        for a, b in zip(before, after):
            prefix_difference += int(b)-int(a)
            assert prefix_difference*first_sign >= 0
    if before_ids[edited_position] == after_ids[edited_position]:
        assert before_ids == after_ids
    return len(changed)


def interpolated_ap_101(tp_flags, fp_flags, ground_truth_count):
    """Abstract 101-point PR-envelope arithmetic; NOT a native COCO witness."""
    assert len(tp_flags) == len(fp_flags) and ground_truth_count > 0
    total_tp = total_fp = 0
    recall, precision = [], []
    for tp, fp in zip(tp_flags, fp_flags):
        assert tp in (0, 1) and fp in (0, 1) and not (tp and fp)
        total_tp += tp
        total_fp += fp
        recall.append(total_tp/ground_truth_count)
        precision.append(total_tp/(total_tp+total_fp+2.220446049250313e-16))
    for index in range(len(precision)-2, -1, -1):
        precision[index] = max(precision[index], precision[index+1])
    answer = 0.0
    for r in range(101):
        k = bisect_left(recall, r/100)
        if k < len(precision):
            answer += precision[k]
    return answer/101


def prefix_dominates(before, after):
    if len(before) != len(after):
        raise ValueError('Fixed candidate identities require equal prefix lengths')
    cumulative = 0
    for a, b in zip(before, after):
        cumulative += b-a
        if cumulative < 0:
            return False
    return True


def merge_local_sequences(first, second, first_positions):
    positions = set(first_positions)
    i = j = 0
    merged = []
    for k in range(len(first)+len(second)):
        if k in positions:
            merged.append(first[i]); i += 1
        else:
            merged.append(second[j]); j += 1
    return merged


def run():
    try:
        prefix_dominates([0], [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError('Unequal-length prefixes must be rejected')
    preferences = [p for size in range(4) for p in permutations(range(3), size)]
    exhaustive_count = maximum_hamming = 0
    for rows in product(preferences, repeat=3):
        for position in range(3):
            for replacement in preferences:
                maximum_hamming = max(maximum_hamming, check_single_row(rows, position, replacement, 3))
                exhaustive_count += 1
    rng = Random(20261102)
    for _ in range(100000):
        m = rng.randrange(1, 15)
        n = rng.randrange(1, 35)
        rows = [tuple(rng.sample(range(m), rng.randrange(m+1))) for _ in range(n)]
        position = rng.randrange(n)
        replacement = tuple(rng.sample(range(m), rng.randrange(m+1)))
        maximum_hamming = max(maximum_hamming, check_single_row(rows, position, replacement, m))

    binary = list(product((0, 1), repeat=3))
    dominance_pairs = [(a, b) for a in binary for b in binary if prefix_dominates(a, b)]
    merges = list(combinations(range(6), 3))
    composed_cases = 0
    for (a0, a1), (b0, b1) in product(dominance_pairs, repeat=2):
        for positions in merges:
            before = merge_local_sequences(a0, b0, positions)
            after = merge_local_sequences(a1, b1, positions)
            assert prefix_dominates(before, after)
            old_ap = interpolated_ap_101(before, [1-x for x in before], 6)
            new_ap = interpolated_ap_101(after, [1-x for x in after], 6)
            assert new_ap >= old_ap-1e-12
            composed_cases += 1

    # Same TP prefixes and same total nonignored count are NOT sufficient.
    # baseline=[ignored,TP,FP], modified=[FP,TP,ignored].
    ignore_before = interpolated_ap_101([0, 1, 0], [0, 0, 1], 1)
    ignore_after = interpolated_ap_101([0, 1, 0], [1, 0, 0], 1)
    assert abs(ignore_before-1.0) < 1e-12
    assert abs(ignore_after-.5) < 1e-12
    assert ignore_after < ignore_before

    # Many GT identities may change while every detection remains a TP.
    rows = [(0, 1), (1, 2), (2, 3), (3, 0)]
    before_ids, before = greedy_match(rows, 4)
    modified = list(rows); modified[0] = (1, 0)
    after_ids, after = greedy_match(modified, 4)
    assert before == after and sum(a != b for a, b in zip(before_ids, after_ids)) == 4

    report = dict(
        status='passed', evaluation_type='simulation_only_mathematical_checks',
        actual_coco_evaluation=False, actual_dataset_access=False,
        unequal_prefix_length_rejected=True,
        python_version=sys.version.split()[0],
        exhaustive=dict(ground_truth_count=3, detection_count=3, partial_preferences=16,
                        row_edits=exhaustive_count),
        random=dict(seed=20261102, row_edits=100000, ground_truth_range=[1, 14], detection_range=[1, 34]),
        maximum_tp_hamming_observed=maximum_hamming,
        checks=['tp_hamming_le_2', 'net_tp_change_abs_le_1', 'two_flips_have_opposite_signs',
                'first_flip_determines_prefix_sign', 'unchanged_own_assignment_implies_unchanged_all_assignments'],
        composition=dict(local_length=3, local_dominance_pairs=len(dominance_pairs),
                         images=2, stable_interleavings=len(merges), cases=composed_cases,
                         ap_arithmetic='abstract 101-point envelope; not native COCO'),
        ignore_counterexample=dict(baseline_ap=ignore_before, modified_ap=ignore_after,
                                   tp_prefixes_equal=True, nonignored_total_equal=True),
        identity_chain_example=dict(changed_gt_assignments=4, changed_tp_flags=0),
        warning='Mathematical checks do not establish empirical AP gains, novelty, learned predictability or a deployment safety guarantee.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
