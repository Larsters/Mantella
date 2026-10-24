"""Command line for the Mantella model evals. Run from the repo root:

    python -m src.evals build-set --name seed --seeds
    python -m src.evals run --set seed --model gpt-4.1-mini --model gpt-5.4-mini --judge-model gpt-5.4 --max-usd 1.00
    python -m src.evals report --run <run_id>

Every paid command (replay, judge, run) prints a cost estimate and stops there unless --yes is given;
it refuses to start when the expected cost is above --max-usd and stops early when actual spend reaches it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

from src.evals import dataset, judge as judge_mod, replay as replay_mod, report as report_mod, seeds, w12
from src.evals.budget import Budget, Estimate, check_and_confirm
from src.evals.common import Pricing, default_data_dir, read_jsonl, run_dir, set_path, sets_dir, write_jsonl, evals_dir
from src.evals.llm import DEFAULT_BASE_URL, openai_client_factory

DEFAULT_MAX_USD = 0.50


def _items_by_id(path: str) -> dict[str, dict[str, Any]]:
    return {item['id']: item for item in read_jsonl(path)}


def _load_run(path: str) -> dict[str, Any]:
    with open(os.path.join(path, 'run.json'), encoding='utf-8') as f:
        return json.load(f)


def _save_run(path: str, run: dict[str, Any]) -> None:
    with open(os.path.join(path, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(run, f, ensure_ascii=False, indent=2)


def cmd_build_set(args: argparse.Namespace) -> int:
    use_seeds = args.seeds or not args.transcripts
    items: list[dict[str, Any]] = []
    if args.transcripts:
        records = dataset.load_transcript_turns(args.data_dir, since=args.since, conversation=args.conversation)
        scenarios = set(args.scenario.split(',')) if args.scenario else None
        selected, stats = dataset.select_turns(records, include_followups=args.include_followups, include_images=args.include_images,
                                               scenarios=scenarios, limit=args.limit, seed=args.seed)
        print(f'Transcripts: {stats}')
        items += selected
    if use_seeds:
        disabled = {a.strip() for a in args.disabled_actions.split(',')} if args.disabled_actions else None
        seed_items = seeds.build_seed_items(model=args.seed_model, disabled_actions=disabled)
        print(f'Seed scenarios: {len(seed_items)}')
        items += seed_items
    if args.trust:
        csv_bios = seeds.load_bios()
        items = [w12.apply_trust(item, csv_bios=csv_bios) for item in items]
        print(f'W12 trust prompts/tools applied to {len(items)} items '
              f'({sum(1 for i in items if i["tags"].get("tools_rebuilt"))} with rebuilt tools)')
    if args.annotations:
        print(f'Annotations matched {w12.apply_annotations(items, args.annotations)} items')
        if args.annotated_only:
            items = [i for i in items if i['source'] != 'transcript' or i.get('expected')]
    if not items:
        print('Nothing selected. Turn on "Save Conversation Transcripts" in Mantella (Other tab) and play, or use --seeds.')
        return 1
    path = set_path(args.data_dir, args.name)
    count = write_jsonl(path, items)
    by_scenario: dict[str, int] = {}
    for item in items:
        by_scenario[item['tags']['scenario']] = by_scenario.get(item['tags']['scenario'], 0) + 1
    print(f'Wrote {count} items to {path} ({", ".join(f"{k}: {v}" for k, v in sorted(by_scenario.items()))})')
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print(f'Sets in {sets_dir(args.data_dir)}:')
    if os.path.isdir(sets_dir(args.data_dir)):
        for name in sorted(os.listdir(sets_dir(args.data_dir))):
            if name.endswith('.jsonl'):
                print(f'  {name[:-6]}  ({sum(1 for _ in read_jsonl(os.path.join(sets_dir(args.data_dir), name)))} items)')
    print(f'Runs in {evals_dir(args.data_dir)}:')
    if os.path.isdir(evals_dir(args.data_dir)):
        for name in sorted(os.listdir(evals_dir(args.data_dir))):
            if os.path.exists(os.path.join(evals_dir(args.data_dir), name, 'run.json')):
                run = _load_run(os.path.join(evals_dir(args.data_dir), name))
                print(f"  {name}  set={run.get('set')} candidates={[c['label'] for c in run.get('candidates', [])]}")
    return 0


def _prepare_run(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Creates (or extends) a run folder with a snapshot of the eval set."""
    if args.run:
        path = run_dir(args.data_dir, args.run)
        run = _load_run(path)
        known = {c['label']: c for c in run['candidates']}
        for candidate in candidates:
            if candidate['label'] in known and known[candidate['label']] != candidate:
                raise SystemExit(f"Candidate '{candidate['label']}' differs from the one stored in this run; use a new label.")
            known[candidate['label']] = candidate
        run['candidates'] = list(known.values())
    else:
        source = set_path(args.data_dir, args.set)
        if not os.path.exists(source):
            raise SystemExit(f'Eval set not found: {source} (create it with build-set)')
        run_id = time.strftime('%Y%m%d-%H%M%S-') + os.path.splitext(os.path.basename(source))[0]
        path = run_dir(args.data_dir, run_id)
        os.makedirs(path, exist_ok=True)
        items = list(read_jsonl(source))
        if args.limit:
            items = items[:args.limit]
        write_jsonl(os.path.join(path, 'items.jsonl'), items)
        run = {'run_id': run_id, 'set': source, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'candidates': candidates,
               'repeats': args.repeats}
    items = list(read_jsonl(os.path.join(path, 'items.jsonl')))
    return path, run, items


def _new_run_cleanup(args: argparse.Namespace, path: str) -> None:
    """A dry run must not leave an empty run folder behind."""
    if not args.run and not os.path.exists(os.path.join(path, 'replies.jsonl')):
        for name in ('items.jsonl', 'run.json'):
            if os.path.exists(os.path.join(path, name)):
                os.remove(os.path.join(path, name))
        try:
            os.rmdir(path)
        except OSError:
            pass


def _judge_params(args: argparse.Namespace) -> dict[str, Any]:
    params = dict(judge_mod.DEFAULT_JUDGE_PARAMS)
    if args.judge_params:
        params.update(json.loads(args.judge_params))
    return {k: v for k, v in params.items() if v is not None}


def _placeholder_replies(items: list[dict[str, Any]], candidates: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    text = 'A placeholder reply of typical length for estimating the judge input size. ' * 3
    return [{'item_id': i['id'], 'candidate': c['label'], 'repeat': r, 'content': text, 'tool_calls': None}
            for r in range(repeats) for i in items for c in candidates]


def cmd_replay(args: argparse.Namespace, client_factory: Callable | None = None, do_judge: bool = False) -> int:
    candidates = replay_mod.load_candidates(args.candidates, args.model)
    pricing = Pricing()
    path, run, items = _prepare_run(args, candidates)
    estimate = replay_mod.estimate_replay(items, candidates, args.repeats, pricing)
    if do_judge:
        params = _judge_params(args)
        estimate.add(judge_mod.estimate_judge({i['id']: i for i in items}, _placeholder_replies(items, candidates, args.repeats),
                                              args.judge_model, params, pricing))
    print(f'Run folder: {path}\nItems: {len(items)} · candidates: {", ".join(c["label"] for c in candidates)} · repeats: {args.repeats}')
    if not check_and_confirm(estimate, args.max_usd, args.yes):
        _new_run_cleanup(args, path)
        return 2
    _save_run(path, run)
    factory = client_factory or openai_client_factory(args.key_file)
    budget = Budget(args.max_usd)
    summary = replay_mod.run_replay(items, candidates, args.repeats, os.path.join(path, 'replies.jsonl'), factory, pricing,
                                    budget, concurrency=args.concurrency, seed=args.seed)
    print(f'Replay: {summary}')
    if do_judge:
        _judge_run(args, path, factory, pricing, budget)
        report = report_mod.write_report(path, _items_by_id(os.path.join(path, 'items.jsonl')), args.judge_model, run['candidates'])
        _print_leaderboard(report, path)
    else:
        print(f'Next: python -m src.evals judge --run {run["run_id"]}')
    return 0


def _judge_run(args: argparse.Namespace, path: str, factory: Callable, pricing: Pricing, budget: Budget) -> dict[str, Any]:
    items = _items_by_id(os.path.join(path, 'items.jsonl'))
    replies = _only_candidates(args, judge_mod.judgeable(judge_mod.latest_records(os.path.join(path, 'replies.jsonl'))))
    summary = judge_mod.run_judge(items, replies, args.judge_model, _judge_params(args), os.path.join(path, 'judgments.jsonl'),
                                  factory, pricing, budget, base_url=args.judge_base_url, api_key_env=args.judge_api_key_env,
                                  concurrency=args.concurrency, seed=args.seed)
    print(f'Judge: {summary}')
    return summary


def _only_candidates(args: argparse.Namespace, replies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = getattr(args, 'candidate', None)
    return [r for r in replies if r['candidate'] in wanted] if wanted else replies


def cmd_judge(args: argparse.Namespace, client_factory: Callable | None = None) -> int:
    path = run_dir(args.data_dir, args.run)
    items = _items_by_id(os.path.join(path, 'items.jsonl'))
    replies = judge_mod.judgeable(judge_mod.latest_records(os.path.join(path, 'replies.jsonl')))
    done = {k[:3] for k, r in judge_mod.latest_records(os.path.join(path, 'judgments.jsonl'),
                                                         ('item_id', 'candidate', 'repeat', 'judge_model')).items()
            if r.get('scores') and k[3] == args.judge_model}
    todo = _only_candidates(args, [r for r in replies if (r['item_id'], r['candidate'], r.get('repeat', 0)) not in done])
    pricing = Pricing()
    estimate = judge_mod.estimate_judge(items, todo, args.judge_model, _judge_params(args), pricing)
    print(f'Run folder: {path}\nReplies to judge: {len(todo)} (already judged: {len(done)})')
    if not todo:
        print('Nothing to judge.')
        return 0
    if not check_and_confirm(estimate, args.max_usd, args.yes):
        return 2
    _judge_run(args, path, client_factory or openai_client_factory(args.key_file), pricing, Budget(args.max_usd))
    print(f'Next: python -m src.evals report --run {os.path.basename(path)}')
    return 0


def _print_leaderboard(report: dict[str, Any], path: str) -> None:
    print(f"\nLeaderboard (judge: {report['judge_model']})")
    print(f"{'candidate':<28} {'overall':>12} " + ' '.join(f'{c[:9]:>9}' for c in report['criteria'])
          + f" {'rules':>6} {'ttft50':>7} {'ttft90':>7} {'$/1k':>7}")
    for row in report['leaderboard']:
        o = row['overall']
        overall = '-' if o['mean'] is None else f"{o['mean']:.2f}±{o['ci95'] or 0:.2f}"
        crit = ' '.join(f"{row['criteria'][c]['mean']:>9.2f}" if row['criteria'][c]['mean'] is not None else f"{'-':>9}" for c in report['criteria'])
        rules = '-' if row['action_rule_pass_rate'] is None else f"{row['action_rule_pass_rate'] * 100:.0f}%"
        cost = '-' if row['cost_per_reply_usd'] is None else f"{row['cost_per_reply_usd'] * 1000:.2f}"
        print(f"{row['candidate']:<28} {overall:>12} {crit} {rules:>6} {str(row['ttft_p50']):>7} {str(row['ttft_p90']):>7} {cost:>7}")
    print(f"\nReport: {os.path.join(path, 'report.html')}")


def cmd_report(args: argparse.Namespace) -> int:
    path = run_dir(args.data_dir, args.run)
    run = _load_run(path)
    report = report_mod.write_report(path, _items_by_id(os.path.join(path, 'items.jsonl')), args.judge_model, run.get('candidates'))
    _print_leaderboard(report, path)
    if args.open:
        os.system(f'open "{os.path.join(path, "report.html")}"')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m src.evals', description='Mantella model evals (offline).')
    parser.add_argument('--data-dir', default=default_data_dir(), help='Mantella data folder (default: %(default)s)')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('build-set', help='Build an eval set from transcripts and/or the seed scenarios (free)')
    p.add_argument('--name', required=True)
    p.add_argument('--seeds', action='store_true', help='Include the 12 hand-written seed scenarios (default if --transcripts is not given)')
    p.add_argument('--transcripts', action='store_true', help='Include recorded turns from data/transcripts')
    p.add_argument('--limit', type=int, help='Max transcript turns (stratified by scenario and tool use)')
    p.add_argument('--since', help='Only turns recorded at or after this ISO time, e.g. 2026-10-24')
    p.add_argument('--conversation', help='Only this conversation id')
    p.add_argument('--scenario', help='Comma list: single,multi,radiant')
    p.add_argument('--include-followups', action='store_true', help='Keep Mantella\'s second (text-only) call after a tool-only reply')
    p.add_argument('--include-images', action='store_true', help='Keep turns whose screenshot was stripped (replayed without it)')
    p.add_argument('--disabled-actions', help='Comma list of action names to leave out of seed tools')
    p.add_argument('--seed-model', default='gpt-4.1-mini', help='Model recorded as the original for seed items')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--trust', action='store_true', help='Make items match the server with living_npcs_v1 + W12 trust_and_honesty_enabled: '
                   'prompt block, trust tool descriptions, rebuilt tools for tool-model transcripts, tool-model system prompt')
    p.add_argument('--annotated-only', action='store_true', help='With --annotations: drop transcript turns that have no label')
    p.add_argument('--annotations', help='JSON {item_id: {notes, expect_tools, forbid_tools}} with hand labels for real turns')

    sub.add_parser('list', help='List eval sets and runs')

    def paid(p: argparse.ArgumentParser) -> None:
        p.add_argument('--max-usd', type=float, default=DEFAULT_MAX_USD, help='Spending cap in USD (default %(default)s)')
        p.add_argument('--yes', action='store_true', help='Actually spend money (otherwise: estimate only)')
        p.add_argument('--concurrency', type=int, default=2)
        p.add_argument('--key-file', help='API key file (default: $OPENAI_API_KEY, ./GPT_SECRET_KEY.txt, ~/Games/mantella-server/GPT_SECRET_KEY.txt)')
        p.add_argument('--seed', type=int, default=7)

    def judge_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument('--judge-model', default=judge_mod.DEFAULT_JUDGE_MODEL)
        p.add_argument('--judge-params', help=f'JSON merged over {json.dumps(judge_mod.DEFAULT_JUDGE_PARAMS)} (null removes a key)')
        p.add_argument('--judge-base-url', default=DEFAULT_BASE_URL)
        p.add_argument('--judge-api-key-env', help='Env var holding the judge API key (default OPENAI_API_KEY / key file)')

    def replay_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument('--set', help='Eval set name or path (required unless --run)')
        p.add_argument('--run', help='Existing run id to resume or extend with more candidates')
        p.add_argument('--model', action='append', help='Candidate model (repeatable; label = model)')
        p.add_argument('--candidates', help='JSON file with candidates (see src/evals/candidates.example.json)')
        p.add_argument('--repeats', type=int, default=1, help='Replies per item and candidate (more = tighter CIs)')
        p.add_argument('--limit', type=int, help='Use only the first N items of the set')

    p = sub.add_parser('replay', help='Re-run the set against candidate models (paid)')
    replay_opts(p)
    paid(p)

    p = sub.add_parser('judge', help='Score replies with an LLM judge (paid)')
    p.add_argument('--run', required=True)
    p.add_argument('--candidate', action='append', help='Judge only this candidate label (repeatable; default: all)')
    judge_opts(p)
    paid(p)

    p = sub.add_parser('run', help='replay + judge + report with one combined estimate and cap (paid)')
    replay_opts(p)
    judge_opts(p)
    paid(p)

    p = sub.add_parser('report', help='Write report.json + report.html for a run (free)')
    p.add_argument('--run', required=True)
    p.add_argument('--judge-model', help='Which judge\'s scores to use (default: the most used)')
    p.add_argument('--open', action='store_true', help='Open the HTML report')
    return parser


def main(argv: list[str] | None = None, client_factory: Callable | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ('replay', 'run') and not args.set and not args.run:
        print('error: --set or --run is required')
        return 1
    if args.command == 'build-set':
        return cmd_build_set(args)
    if args.command == 'list':
        return cmd_list(args)
    if args.command == 'replay':
        return cmd_replay(args, client_factory)
    if args.command == 'run':
        return cmd_replay(args, client_factory, do_judge=True)
    if args.command == 'judge':
        return cmd_judge(args, client_factory)
    if args.command == 'report':
        return cmd_report(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
