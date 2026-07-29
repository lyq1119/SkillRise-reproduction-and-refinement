#!/usr/bin/env python3
"""Offline recomputation of WebShop's official 0~1 partial-match task_score from
saved trajectory logs (traj_logs/{train,val}/rollout_*.jsonl).

The traj logs only store the rule-based 0/10 reward, not the raw 0~1 score. But
each trial records the goal instruction text, the purchased asin, and the option
clicks. That is enough to call the env's original get_reward() and recover the
0~1 score exactly.

Run with Java 11 on PATH so spaCy/thefuzz-based reward matches the env:
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-... python recompute_webshop_score.py <traj_dir>
"""
import sys
import os
import re
import json
import argparse
import random

import numpy as np

WEBSHOP_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'agent_system/environments/webshop/webshop',
)
sys.path.insert(0, WEBSHOP_ROOT)
DATA_DIR = os.path.join(WEBSHOP_ROOT, 'data')

from web_agent_site.engine.goal import get_goals, get_reward  # noqa: E402
from web_agent_site.engine.engine import load_products  # noqa: E402

# Non-option clickables that must never be treated as an option selection.
_NON_OPTION_CLICKS = {
    'buy now', 'search', '< prev', 'next', 'back to search',
    'features', 'reviews', 'description', 'attributes', '< prev page', 'next page',
}
_ASIN_RE = re.compile(r'^[a-z0-9]{10}$')
_ACTION_RE = re.compile(r'<action>\s*(.*?)\s*</action>', re.S)
_CLICK_RE = re.compile(r'click\[(.+)\]', re.S)
_GOAL_RE = re.compile(r'Your task is to:\s*(.+?)\n', re.S)


def build_lookups():
    """Deterministic lookups (no RNG): products + a product-level instruction
    prefix map (instruction text WITHOUT options/price) -> asin. The goal's
    product-level fields (query/name/product_category/Attributes) come from the
    product dict; options/price are parsed from the trajectory goal text. This
    avoids reproducing the env's randomized goal sampling entirely."""
    fp = os.path.join(DATA_DIR, 'items_shuffle_1000.json')
    ap = os.path.join(DATA_DIR, 'items_ins_v2_1000.json')
    all_products, product_item_dict, product_prices, _ = load_products(
        filepath=fp, attrpath=ap, num_products=1000, human_goals=False)
    with open(ap) as f:
        ins = json.load(f)
    base2asin = {}
    asin2instr_attrs = {}
    for asin, v in ins.items():
        it = v.get('instruction')
        if it:
            base2asin.setdefault(it.strip().rstrip('.'), asin)
        # goal['attributes'] in synthetic goals = product['instruction_attributes']
        asin2instr_attrs[asin] = v.get('instruction_attributes', v.get('attributes', []))
    # longest base first so startswith picks the most specific instruction
    base_keys = sorted(base2asin.keys(), key=len, reverse=True)
    return product_item_dict, product_prices, base2asin, base_keys, asin2instr_attrs


_PRICE_RE = re.compile(r', and price lower than ([\d.]+) dollars')


def parse_goal_fields(goal_text, base2asin, base_keys, product_item_dict,
                      asin2instr_attrs):
    """Locate the goal's target asin by instruction prefix, then build a goal dict
    with options/price parsed from the text. Returns goal dict or None."""
    t = goal_text.strip()
    asin = None
    base = None
    for k in base_keys:
        if t.startswith(k):
            asin = base2asin[k]
            base = k
            break
    if asin is None or asin not in product_item_dict:
        return None
    prod = product_item_dict[asin]
    # price_upper from text
    price = None
    pm = _PRICE_RE.search(t)
    if pm:
        price = float(pm.group(1))
    # options: the text tail after `base` is ' with <opt: val>, and ... [price]'.
    # The product's own option categories tell us exactly which key each value is.
    goal_options = {}
    tail = t[len(base):]
    if pm:
        tail = tail[:tail.rfind(', and price lower than')] if ', and price lower than' in tail else tail
    prod_opts = prod.get('options', {})
    val2cat = {}
    for cat, vals in prod_opts.items():
        for v in vals:
            val2cat[v.lower()] = cat.lower()
    # parse 'key: value' pairs from tail; map value back to a real option category
    for km in re.finditer(r': ([^,]+?)(?=, and |$)', tail):
        v = km.group(1).strip().rstrip('.')
        cat = val2cat.get(v.lower())
        if cat is not None:
            goal_options[cat] = v
    goal = {
        'asin': asin,
        'category': prod.get('category'),
        'query': prod.get('query'),
        'name': prod.get('name'),
        'product_category': prod.get('product_category'),
        'instruction_text': goal_text.strip(),
        'attributes': asin2instr_attrs.get(asin, []),
        'price_upper': price if price is not None else 1000000,
        'goal_options': goal_options,
    }
    return goal


def parse_attempt(steps, product_item_dict):
    """From ONE attempt's play steps (single task_pos), recover (goal_text, asin,
    options dict). Mirrors the env: asin = last product-link click, options keyed
    by the product's option category with later clicks overwriting earlier ones."""
    goal_text = None
    asin = None
    raw_option_clicks = []
    for s in steps:
        if goal_text is None:
            m = _GOAL_RE.search(s.get('input', ''))
            if m:
                goal_text = m.group(1)
        for a in _ACTION_RE.findall(s.get('response', '')):
            cm = _CLICK_RE.match(a.strip().lower())
            if not cm:
                continue
            val = cm.group(1).strip()
            if _ASIN_RE.match(val):
                asin = val.upper()
            elif val in _NON_OPTION_CLICKS:
                continue
            else:
                raw_option_clicks.append(val)
    options = {}
    if asin and asin in product_item_dict:
        prod_opts = product_item_dict[asin].get('options', {})
        val2cat = {}
        for cat, vals in prod_opts.items():
            for v in vals:
                val2cat[v.lower()] = cat.lower()
        for v in raw_option_clicks:
            cat = val2cat.get(v)
            if cat is not None:
                options[cat] = v  # later click overwrites (matches env behavior)
    return goal_text, asin, options


def split_attempts(steps, num_attempts):
    """Group a trial-group's play steps by task_pos into per-attempt step lists."""
    by_pos = {p: [] for p in range(num_attempts)}
    for s in steps:
        if s.get('phase') != 'play':
            continue
        pos = s.get('task_pos')
        if pos in by_pos:
            by_pos[pos].append(s)
    return [by_pos[p] for p in range(num_attempts)]


def score_attempt(att_steps, lookups, stats):
    """Recompute 0~1 task_score for a single attempt. Returns (score, located)."""
    product_item_dict, product_prices, base2asin, base_keys, asin2instr_attrs = lookups
    if not att_steps:
        return 0.0, True  # attempt absent -> treat as no purchase, score 0
    goal_text, bought_asin, options = parse_attempt(att_steps, product_item_dict)
    if goal_text is None:
        stats['no_goal'] += 1
        return 0.0, False
    g = parse_goal_fields(goal_text, base2asin, base_keys, product_item_dict,
                          asin2instr_attrs)
    if g is None:
        stats['no_goal'] += 1
        return 0.0, False
    if bought_asin is None or bought_asin not in product_item_dict:
        stats['no_asin'] += 1
        return 0.0, True  # no purchase -> score 0
    pp = product_item_dict[bought_asin]
    price = product_prices.get(bought_asin)
    r = float(get_reward(pp, g, price, options))
    stats['scored'] += 1
    return r, True


def recompute_file(path, lookups, num_attempts=3):
    """Each jsonl line is one trial GROUP of K=num_attempts attempts (same task,
    retried). Per base.py, task_score[k]/success_rate[k] are CUMULATIVE best over
    attempts 0..k. success_rate uses the env-stored 0/10 reward (ground truth);
    task_score is recomputed from the trajectory."""
    cum_score = [0.0] * num_attempts
    cum_won = [0.0] * num_attempts
    n_groups = 0
    stats = {'no_goal': 0, 'no_asin': 0, 'no_product': 0, 'scored': 0, 'located': 0,
             'attempts': 0}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            attempts = split_attempts(d.get('steps', []), num_attempts)
            # ground-truth won per attempt from stored 0/10 reward field
            try:
                rw = eval(d.get('reward', '[]'))
            except Exception:
                rw = []
            per_score = [0.0] * num_attempts
            per_won = [False] * num_attempts
            for k in range(num_attempts):
                sc, located = score_attempt(attempts[k], lookups, stats)
                per_score[k] = sc
                stats['attempts'] += 1
                if located:
                    stats['located'] += 1
                per_won[k] = (k < len(rw) and float(rw[k]) == 10.0)
            best_s = 0.0
            best_w = False
            for k in range(num_attempts):
                best_s = max(best_s, per_score[k])
                best_w = best_w or per_won[k]
                cum_score[k] += best_s
                cum_won[k] += 1.0 if best_w else 0.0
            n_groups += 1
    mean_score = [cum_score[k] / n_groups if n_groups else float('nan')
                  for k in range(num_attempts)]
    mean_won = [cum_won[k] / n_groups if n_groups else float('nan')
                for k in range(num_attempts)]
    return mean_score, mean_won, n_groups, stats


def step_from_filename(path):
    m = re.search(r'rollout_(\d+)\.jsonl', os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traj_dir', help='directory of rollout_*.jsonl (e.g. .../traj_logs/val)')
    ap.add_argument('--cadence', type=int, default=6,
                    help='val dump spacing (test_freq+1 with val_before_train); '
                         'step = dump*test_freq/cadence')
    ap.add_argument('--test_freq', type=int, default=5)
    args = ap.parse_args()

    lookups = build_lookups()
    pid, _, base2asin, _, _ = lookups
    print(f'Loaded {len(pid)} products, {len(base2asin)} base instructions',
          file=sys.stderr)

    files = sorted(
        [os.path.join(args.traj_dir, f) for f in os.listdir(args.traj_dir)
         if f.startswith('rollout_') and f.endswith('.jsonl')],
        key=step_from_filename,
    )
    K = 3
    hdr = ('step\tdump\tn\tlocated%\t'
           + '\t'.join(f'score[{k}]' for k in range(K)) + '\t'
           + '\t'.join(f'sr[{k}]' for k in range(K)))
    print(hdr)
    all_rows = []
    for f in files:
        mean_score, mean_won, n, st = recompute_file(f, lookups, K)
        dump = step_from_filename(f)
        step = round(dump * args.test_freq / args.cadence)
        loc = 100.0 * st['located'] / st['attempts'] if st['attempts'] else 0.0
        all_rows.append((step, dump, n, mean_score, mean_won, loc))
        print(f'{step}\t{dump}\t{n}\t{loc:.1f}\t'
              + '\t'.join(f'{s:.4f}' for s in mean_score) + '\t'
              + '\t'.join(f'{w:.4f}' for w in mean_won))
    return all_rows


if __name__ == '__main__':
    main()
