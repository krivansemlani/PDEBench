import json
import re
import argparse
from collections import defaultdict

# ============================================================
# CONSTANTS
# ============================================================

FAMILIES = ['Burgers', 'Wave', 'Laplace', 'KleinGordon', 'Heat', 'Advection']
OPERATORS = ['exp', 'sin', 'cos', 'tanh', 'polynomial']
DIALECTS = ['latex', 'prefix', 'postfix', 'natural']

# Keywords that should appear in reasoning for each family.
# Trash Score fires when family prediction is correct but none of these match.
REASONING_KEYWORDS = {
    'Burgers':     ['nonlinear', 'convect', 'viscosit', 'shock', 'u*u', 'transport', 'burger'],
    'Wave':        ['wave', 'propagat', 'u_tt', 'second-order time', 'speed', 'oscillat', 'hyperbolic'],
    'Laplace':     ['steady', 'equilibrium', 'no time', 'elliptic', 'harmonic', 'laplace'],
    'KleinGordon': ['mass', 'klein', 'gordon', 'relativistic', 'dispersive', 'u_tt'],
    'Heat':        ['diffus', 'thermal', 'u_t', 'first-order time', 'parabolic', 'heat'],
    'Advection':   ['advect', 'transport', 'first-order', 'u_t', 'u_x', 'u_y', 'u_z'],
}


# ============================================================
# PARSER
# ============================================================

def parse_output(text):
    """
    Extract (family, operators, reasoning) from a model output string.

    Expected format: "family: X | operators: op1, op2 | reasoning: text"
    Falls back to loose regex when the model drifts from this format.

    Returns:
        family    : str or None (None = parse failure)
        operators : list of str (may be empty)
        reasoning : str (may be empty)
    """
    if not isinstance(text, str):
        return None, [], ""

    family = None
    operators = []
    reasoning = ""

    # --- family ---
    family_match = re.search(r'family\s*:\s*(\w+)', text, re.IGNORECASE)
    if family_match:
        raw = family_match.group(1).strip()
        for f in FAMILIES:
            if raw.lower() == f.lower():
                family = f
                break

    # --- operators ---
    ops_match = re.search(r'operators\s*:\s*([^|]+)', text, re.IGNORECASE)
    if ops_match:
        ops_raw = ops_match.group(1)
        for op in OPERATORS:
            if re.search(rf'\b{op}\b', ops_raw, re.IGNORECASE):
                operators.append(op)

    # --- reasoning ---
    reasoning_match = re.search(r'reasoning\s*:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return family, operators, reasoning


# ============================================================
# METRICS
# ============================================================

def compute_family_accuracy(pred_families, true_families):
    """Fraction of instances where predicted family == ground-truth family."""
    assert len(pred_families) == len(true_families)
    correct = sum(p == t for p, t in zip(pred_families, true_families))
    return correct / len(pred_families)


def compute_operator_f1(pred_ops_list, true_ops_list):
    """
    Macro-averaged set-level F1 across all instances.

    For each instance: precision = |pred ∩ true| / |pred|
                       recall    = |pred ∩ true| / |true|
                       f1        = harmonic mean
    """
    assert len(pred_ops_list) == len(true_ops_list)
    f1s = []
    for pred, true in zip(pred_ops_list, true_ops_list):
        pred_set = set(pred)
        true_set = set(true)
        if not pred_set and not true_set:
            f1s.append(1.0)
            continue
        if not pred_set or not true_set:
            f1s.append(0.0)
            continue
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set)
        recall    = tp / len(true_set)
        denom = precision + recall
        f1s.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return sum(f1s) / len(f1s)


def compute_trash_score(pred_families, true_families, reasonings):
    """
    Among instances where family prediction is correct,
    fraction whose reasoning contains no expected keywords for that family.

    High trash score = model gets family right by surface pattern-matching,
    not by genuine physical reasoning.
    """
    assert len(pred_families) == len(true_families) == len(reasonings)

    correct_indices = [
        i for i, (p, t) in enumerate(zip(pred_families, true_families))
        if p is not None and p == t
    ]
    if not correct_indices:
        return 0.0, 0  # score, denominator

    trash = 0
    for i in correct_indices:
        family = true_families[i]
        keywords = REASONING_KEYWORDS.get(family, [])
        reasoning_lower = reasonings[i].lower()
        if not any(kw.lower() in reasoning_lower for kw in keywords):
            trash += 1

    return trash / len(correct_indices), len(correct_indices)


# ============================================================
# EVALUATION RUNNER
# ============================================================

def evaluate(records):
    """
    Compute all metrics given a list of records.

    Each record must have:
        'prediction' : raw model output string
        'target'     : ground-truth output string
        'family'     : ground-truth family label (str)
        'dialect'    : input dialect used (str)  [optional, for per-dialect breakdown]
    """
    pred_families = []
    pred_ops_list = []
    reasonings    = []
    true_families = []
    true_ops_list = []
    parse_failures = 0

    for r in records:
        pf, po, pr = parse_output(r['prediction'])
        _, go, _   = parse_output(r['target'])

        if pf is None:
            parse_failures += 1

        pred_families.append(pf)
        pred_ops_list.append(po)
        reasonings.append(pr)
        true_families.append(r['family'])
        true_ops_list.append(go)

    n = len(records)
    fam_acc   = compute_family_accuracy(pred_families, true_families)
    op_f1     = compute_operator_f1(pred_ops_list, true_ops_list)
    trash, n_correct = compute_trash_score(pred_families, true_families, reasonings)

    results = {
        'n': n,
        'parse_failure_rate': parse_failures / n,
        'family_accuracy':    fam_acc,
        'operator_f1':        op_f1,
        'trash_score':        trash,
        'n_correct_family':   n_correct,
    }

    # --- per-family breakdown ---
    by_family = defaultdict(list)
    for i, r in enumerate(records):
        by_family[r['family']].append(i)

    results['per_family'] = {}
    for fam, indices in by_family.items():
        pf_sub = [pred_families[i] for i in indices]
        po_sub = [pred_ops_list[i] for i in indices]
        re_sub = [reasonings[i] for i in indices]
        tf_sub = [true_families[i] for i in indices]
        to_sub = [true_ops_list[i] for i in indices]
        ts, nc = compute_trash_score(pf_sub, tf_sub, re_sub)
        results['per_family'][fam] = {
            'n':               len(indices),
            'family_accuracy': compute_family_accuracy(pf_sub, tf_sub),
            'operator_f1':     compute_operator_f1(po_sub, to_sub),
            'trash_score':     ts,
        }

    # --- per-dialect breakdown (if dialect field present) ---
    if all('dialect' in r for r in records):
        by_dialect = defaultdict(list)
        for i, r in enumerate(records):
            by_dialect[r['dialect']].append(i)

        results['per_dialect'] = {}
        for dial, indices in by_dialect.items():
            pf_sub = [pred_families[i] for i in indices]
            po_sub = [pred_ops_list[i] for i in indices]
            re_sub = [reasonings[i] for i in indices]
            tf_sub = [true_families[i] for i in indices]
            to_sub = [true_ops_list[i] for i in indices]
            ts, nc = compute_trash_score(pf_sub, tf_sub, re_sub)
            results['per_dialect'][dial] = {
                'n':               len(indices),
                'family_accuracy': compute_family_accuracy(pf_sub, tf_sub),
                'operator_f1':     compute_operator_f1(po_sub, to_sub),
                'trash_score':     ts,
            }

    return results


# ============================================================
# PRETTY PRINT
# ============================================================

def print_results(results, label=""):
    header = f"=== Results{': ' + label if label else ''} ==="
    print(header)
    print(f"  N instances:        {results['n']}")
    print(f"  Parse failure rate: {results['parse_failure_rate']:.3f}")
    print(f"  Family accuracy:    {results['family_accuracy']:.3f}")
    print(f"  Operator F1:        {results['operator_f1']:.3f}")
    print(f"  Trash score:        {results['trash_score']:.3f}  (over {results['n_correct_family']} correct-family preds)")

    if 'per_dialect' in results:
        print("\n  Per-dialect breakdown:")
        print(f"  {'Dialect':<12} {'FamAcc':>8} {'OpF1':>8} {'Trash':>8}")
        print("  " + "-" * 42)
        for dial in DIALECTS:
            if dial in results['per_dialect']:
                d = results['per_dialect'][dial]
                print(f"  {dial:<12} {d['family_accuracy']:>8.3f} {d['operator_f1']:>8.3f} {d['trash_score']:>8.3f}")

    if 'per_family' in results:
        print("\n  Per-family breakdown:")
        print(f"  {'Family':<14} {'FamAcc':>8} {'OpF1':>8} {'Trash':>8}")
        print("  " + "-" * 46)
        for fam in FAMILIES:
            if fam in results['per_family']:
                d = results['per_family'][fam]
                print(f"  {fam:<14} {d['family_accuracy']:>8.3f} {d['operator_f1']:>8.3f} {d['trash_score']:>8.3f}")
    print()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate PDE dialect model predictions.")
    parser.add_argument('--predictions', required=True,
                        help='JSONL file with one {"prediction": ..., "target": ..., "family": ..., "dialect": ...} per line')
    parser.add_argument('--label', default='', help='Optional label for the printed results header')
    args = parser.parse_args()

    records = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results = evaluate(records)
    print_results(results, label=args.label)

    # also save results as json alongside predictions file
    out_path = args.predictions.replace('.jsonl', '_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == '__main__':
    main()
