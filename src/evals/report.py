"""Leaderboard per candidate: judge scores (mean ± 95% CI), deterministic checks, latency percentiles, cost.
Writes report.json and a self-contained report.html (no external assets; light/dark)."""
from __future__ import annotations

import html
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any

from src.evals.judge import CRITERIA, latest_records

# two-sided 95% t critical values, df 1..30
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': None, 'ci95': None}
    mean = sum(values) / n
    if n == 1:
        return {'n': 1, 'mean': round(mean, 3), 'ci95': None}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    t = _T95[n - 2] if n - 1 <= 30 else 1.96
    return {'n': n, 'mean': round(mean, 3), 'ci95': round(t * sd / math.sqrt(n), 3)}


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)


def _rate(flags: list[bool]) -> float | None:
    return round(sum(1 for f in flags if f) / len(flags), 3) if flags else None


def pick_judge(judgments: list[dict[str, Any]], judge_model: str | None) -> str | None:
    if judge_model:
        return judge_model
    counts = Counter(j['judge_model'] for j in judgments if j.get('scores'))
    return counts.most_common(1)[0][0] if counts else None


def aggregate(items: dict[str, dict[str, Any]], replies_path: str, judgments_path: str,
              judge_model: str | None = None, candidates_meta: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    replies = latest_records(replies_path)
    all_judgments = list(latest_records(judgments_path, ('item_id', 'candidate', 'repeat', 'judge_model')).values())
    judge = pick_judge(all_judgments, judge_model)
    judgments = {(j['item_id'], j['candidate'], j.get('repeat', 0)): j for j in all_judgments if j.get('judge_model') == judge}
    meta = {c['label']: c for c in candidates_meta or []}

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reply in replies.values():
        by_candidate[reply['candidate']].append(reply)

    rows = []
    for label, cand_replies in by_candidate.items():
        answered = [r for r in cand_replies if not r.get('skipped')]
        ok = [r for r in answered if not r.get('error')]
        scored = [(r, judgments.get((r['item_id'], r['candidate'], r.get('repeat', 0)))) for r in ok]
        scored = [(r, j) for r, j in scored if j and j.get('scores')]
        criteria = {c: mean_ci([j['scores'][c] for _, j in scored]) for c in CRITERIA}
        overall = mean_ci([sum(j['scores'][c] for c in CRITERIA) / len(CRITERIA) for _, j in scored])
        checks = [r['action_check'] for r in ok if r.get('action_check')]
        costs = [r['cost_usd'] for r in ok if r.get('cost_usd') is not None]
        voices = [r.get('voice') or {} for r in ok]
        by_scenario: dict[str, list[float]] = defaultdict(list)
        for r, j in scored:
            scenario = items.get(r['item_id'], {}).get('tags', {}).get('scenario', '?')
            by_scenario[scenario].append(sum(j['scores'][c] for c in CRITERIA) / len(CRITERIA))
        rows.append({
            'candidate': label,
            'model': cand_replies[0].get('model'),
            'params': meta.get(label, {}).get('params') or {},
            'replies': len(answered), 'errors': len(answered) - len(ok), 'judged': len(scored),
            'skipped': len(cand_replies) - len(answered),
            'overall': overall, 'criteria': criteria,
            'by_scenario': {s: mean_ci(v) for s, v in sorted(by_scenario.items())},
            'action_rule_pass_rate': _rate([c['pass'] for c in checks]), 'action_rule_n': len(checks),
            'forbidden_tool_rate': _rate([bool(c['forbidden_called']) for c in checks]),
            'tool_call_rate': _rate([bool(r.get('tool_calls')) for r in ok]),
            'tool_only_rate': _rate([v.get('tool_only', False) for v in voices]),
            'voice_flags_rate': _rate([any(v.get(k) for k in ('markdown', 'asterisks', 'quotes', 'name_prefix_in_single', 'over_sentence_limit')) for v in voices]),
            'mean_words': round(sum(v.get('words', 0) for v in voices) / len(voices), 1) if voices else None,
            'ttft_p50': percentile([r.get('ttft_s') for r in ok], 50), 'ttft_p90': percentile([r.get('ttft_s') for r in ok], 90),
            'total_p50': percentile([r.get('total_s') for r in ok], 50), 'total_p90': percentile([r.get('total_s') for r in ok], 90),
            'cost_per_reply_usd': round(sum(costs) / len(costs), 6) if costs else None,
            'replay_cost_usd': round(sum(costs), 4),
            'mean_completion_tokens': round(sum((r.get('usage') or {}).get('completion_tokens', 0) for r in ok) / len(ok), 1) if ok else None,
        })
    rows.sort(key=lambda r: (r['overall']['mean'] is None, -(r['overall']['mean'] or 0), r['ttft_p50'] or 1e9))
    judge_cost = round(sum(j.get('cost_usd') or 0 for j in all_judgments), 4)

    details = []
    for (item_id, label, repeat), reply in sorted(replies.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2])):
        if reply.get('skipped'):
            continue
        judgment = judgments.get((item_id, label, repeat)) or {}
        details.append({'item_id': item_id, 'candidate': label, 'repeat': repeat,
                        'content': reply.get('content'), 'tool_calls': [c['function']['name'] + '(' + c['function'].get('arguments', '') + ')' for c in reply.get('tool_calls') or []],
                        'error': reply.get('error'), 'ttft_s': reply.get('ttft_s'), 'total_s': reply.get('total_s'),
                        'cost_usd': reply.get('cost_usd'), 'scores': judgment.get('scores'), 'rationale': judgment.get('rationale'),
                        'action_check': reply.get('action_check')})
    return {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'judge_model': judge, 'judge_cost_usd': judge_cost,
            'criteria': list(CRITERIA), 'leaderboard': rows, 'details': details,
            'items': {k: {'tags': v.get('tags'), 'last_user': _last_user(v), 'expected': (v.get('expected') or {}).get('notes')} for k, v in items.items()}}


def _last_user(item: dict[str, Any]) -> str:
    for message in reversed(item['request']['messages']):
        if message.get('role') == 'user':
            content = message.get('content')
            return content if isinstance(content, str) else json.dumps(content)[:300]
    return ''


# ---------------------------------------------------------------- HTML

def _fmt_score(stat: dict[str, Any] | None) -> str:
    if not stat or stat.get('mean') is None:
        return '<span class="muted">–</span>'
    ci = f'<span class="ci">±{stat["ci95"]:.2f}</span>' if stat.get('ci95') is not None else ''
    width = max(0.0, min(100.0, (stat['mean'] - 1) / 4 * 100))
    return f'<div class="score"><span class="bar" style="width:{width:.0f}%"></span><b>{stat["mean"]:.2f}</b>{ci}</div>'


def _fmt(value: Any, unit: str = '', digits: int = 2) -> str:
    if value is None:
        return '<span class="muted">–</span>'
    if isinstance(value, float):
        return f'{value:.{digits}f}{unit}'
    return f'{value}{unit}'


def _pct(value: float | None) -> str:
    return '<span class="muted">–</span>' if value is None else f'{value * 100:.0f}%'


def render_html(report: dict[str, Any], title: str) -> str:
    e = html.escape
    rows = report['leaderboard']
    head = ''.join(f'<th>{e(c.replace("_", " "))}</th>' for c in report['criteria'])
    body = []
    for rank, row in enumerate(rows, 1):
        params = ', '.join(f'{k}={v}' for k, v in row['params'].items())
        cost = row['cost_per_reply_usd']
        cost_html = '<span class="muted">–</span>' if cost is None else f'${cost * 1000:.2f}'
        errors_html = f'<div class="bad">{row["errors"]} err</div>' if row['errors'] else ''
        params_html = (' · ' + e(params)) if params else ''
        body.append(
            f'<tr><td class="num">{rank}</td><td><b>{e(row["candidate"])}</b><div class="muted small">{e(row["model"] or "")}{params_html}</div></td>'
            f'<td>{_fmt_score(row["overall"])}</td>' + ''.join(f'<td>{_fmt_score(row["criteria"][c])}</td>' for c in report['criteria']) +
            f'<td class="num">{_pct(row["action_rule_pass_rate"])}<div class="muted small">n={row["action_rule_n"]}</div></td>'
            f'<td class="num">{_fmt(row["ttft_p50"], "s")}<div class="muted small">p90 {_fmt(row["ttft_p90"], "s")}</div></td>'
            f'<td class="num">{_fmt(row["total_p50"], "s")}<div class="muted small">p90 {_fmt(row["total_p90"], "s")}</div></td>'
            f'<td class="num">{cost_html}</td>'
            f'<td class="num">{_pct(row["tool_only_rate"])}</td><td class="num">{_pct(row["voice_flags_rate"])}</td>'
            f'<td class="num">{row["judged"]}/{row["replies"]}{errors_html}</td></tr>')
    scenarios = sorted({s for row in rows for s in row['by_scenario']})
    scen_rows = ''.join(
        f'<tr><td><b>{e(row["candidate"])}</b></td>' + ''.join(f'<td>{_fmt_score(row["by_scenario"].get(s))}</td>' for s in scenarios) + '</tr>'
        for row in rows)
    scen_head = ''.join(f'<th>{e(s)}</th>' for s in scenarios)
    detail_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in report['details']:
        detail_by_item[d['item_id']].append(d)
    detail_html = []
    for item_id, entries in detail_by_item.items():
        info = report['items'].get(item_id, {})
        tags = info.get('tags') or {}
        lines = []
        for d in entries:
            scores = ' '.join(f'{k.split("_")[0]} {v}' for k, v in (d['scores'] or {}).items()) or ('error' if d['error'] else 'not judged')
            check = d.get('action_check')
            check_html = '' if not check else (' <span class="ok">rule ok</span>' if check['pass'] else ' <span class="bad">rule fail</span>')
            lines.append(
                f'<div class="reply"><div class="rhead"><b>{e(d["candidate"])}</b> <span class="muted small">{e(scores)} · '
                f'ttft {_fmt(d["ttft_s"], "s")} · total {_fmt(d["total_s"], "s")}</span>{check_html}</div>'
                f'<div class="say">{e(d["content"] or "") or "<span class=muted>(no text)</span>"}</div>'
                + (f'<div class="tools">{e(", ".join(d["tool_calls"]))}</div>' if d['tool_calls'] else '')
                + (f'<div class="bad small">{e(d["error"])}</div>' if d['error'] else '')
                + (f'<div class="muted small">{e(d["rationale"] or "")}</div>' if d.get('rationale') else '') + '</div>')
        detail_html.append(
            f'<details><summary><b>{e(item_id)}</b> <span class="tag">{e(str(tags.get("scenario", "")))}</span> '
            f'<span class="muted">{e((info.get("last_user") or "")[:160])}</span></summary>'
            + (f'<div class="expected"><b>Expected:</b> {e(info["expected"])}</div>' if info.get('expected') else '')
            + ''.join(lines) + '</details>')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title>
<style>
:root{{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e3e1da;--accent:#2f6f5e;--bar:#cfe5dc;--bad:#b3261e;--ok:#2f6f5e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151614;--fg:#ecebe6;--muted:#9a9890;--card:#1e1f1c;--line:#33342f;--accent:#7cc4ab;--bar:#2b4a40;--bad:#f2867d;--ok:#7cc4ab}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1300px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 8px}}
.muted{{color:var(--muted)}}.small{{font-size:12px}}.ci{{color:var(--muted);font-size:11px;margin-left:4px}}
.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:12px;font-weight:600;color:var(--muted);text-transform:capitalize;white-space:nowrap}}td.num{{text-align:right;white-space:nowrap}}
.score{{position:relative;min-width:92px;padding:1px 4px;white-space:nowrap}}.score .bar{{position:absolute;left:0;top:0;bottom:0;background:var(--bar);border-radius:3px;z-index:0}}
.score b,.score .ci{{position:relative;z-index:1}}.bad{{color:var(--bad)}}.ok{{color:var(--ok)}}
details{{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0;padding:8px 12px}}summary{{cursor:pointer}}
.tag{{font-size:11px;border:1px solid var(--line);border-radius:8px;padding:0 6px;color:var(--muted)}}
.reply{{border-top:1px solid var(--line);padding:8px 0}}.say{{white-space:pre-wrap;margin:2px 0}}.tools{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--accent)}}
.expected{{margin:8px 0;padding:6px 10px;border-left:3px solid var(--accent)}}ul{{padding-left:18px}}
</style></head><body><main>
<h1>{e(title)}</h1>
<div class="muted">Generated {e(report["generated"])} · judge: <b>{e(str(report["judge_model"]))}</b> · judge cost ${report["judge_cost_usd"]:.3f}</div>
<h2>Leaderboard</h2>
<div class="wrap"><table><thead><tr><th>#</th><th>Candidate</th><th>Overall</th>{head}<th>Action rules</th><th>TTFT p50</th><th>Total p50</th><th>Cost / 1k replies</th><th>Tool-only</th><th>Voice flags</th><th>Judged</th></tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<ul class="muted small">
<li>Scores are 1–5 from the judge (mean ± 95% CI over replies). Overall = mean of the four criteria. Speed and cost are measured, not judged.</li>
<li>Action rules: share of seed replies that called an expected tool and no forbidden tool (deterministic, no judge).</li>
<li>Tool-only: replies with tool calls but no speech. In the game Mantella then makes a second call for the spoken line (adds about one more round trip).</li>
<li>Voice flags: markdown, asterisks, quotes, a "Name:" prefix in one-on-one talk, or more sentences than Mantella keeps (4 single / 12 multi).</li>
<li>Cost per 1,000 replies in USD, from real token usage incl. cached-input discounts (replay only; judge cost is shown above).</li>
</ul>
<h2>Overall by scenario</h2>
<div class="wrap"><table><thead><tr><th>Candidate</th>{scen_head}</tr></thead><tbody>{scen_rows}</tbody></table></div>
<h2>Replies</h2>
{"".join(detail_html)}
</main></body></html>'''


def write_report(run_path: str, items: dict[str, dict[str, Any]], judge_model: str | None = None,
                 candidates_meta: list[dict[str, Any]] | None = None, title: str | None = None) -> dict[str, Any]:
    report = aggregate(items, os.path.join(run_path, 'replies.jsonl'), os.path.join(run_path, 'judgments.jsonl'),
                       judge_model, candidates_meta)
    report['run'] = os.path.basename(run_path.rstrip(os.sep))
    with open(os.path.join(run_path, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_path, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(render_html(report, title or f'Mantella model evals · {report["run"]}'))
    return report
