"""Export archived replay predictions, joined to labels by episode_key.

Run with --check to verify the supplied CSV without writing any files.
Only the Python standard library is required; no inference is performed.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]
FIELDS = ['configuration', 'run', 'episode_key', 'llm_category',
          'llm_confidence', 'llm_evidence', 'llm_macro']
CONFIGS = ['full', 'no_hints', 'no_weather', 'no_baseline', 'minimal', 'qwen_full']


def replay_predictions(path):
    # Each archived replay is sequential: one input followed by one output.
    pending = None
    predictions = {}
    for line in path.read_text(encoding='utf-16').lstrip('\ufeff').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        data = record['data']
        if data['type'] == 'llm.prediction.input':
            if pending is not None:
                raise ValueError(f'{path}: input without preceding output')
            text = data['input'].split('Context JSON:', 1)[1].lstrip()
            context, _ = json.JSONDecoder().raw_decode(text)
            event = context['event']
            start = datetime.fromisoformat(event['start_time'])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            pending = event['anomaly_type'] + '@' + start.astimezone(timezone.utc).isoformat()
        elif data['type'] == 'llm.prediction.output':
            if pending is None or pending in predictions:
                raise ValueError(f'{path}: unpaired output or duplicate episode')
            predictions[pending] = json.loads(data['output'])
            pending = None
    if pending is not None:
        raise ValueError(f'{path}: missing final output')
    return predictions


def build():
    with (RESULTS / 'labels.csv').open(encoding='utf-8', newline='') as handle:
        labels = list(csv.DictReader(handle))
    keys = {row['episode_key'] for row in labels}
    if len(keys) != len(labels):
        raise ValueError('Duplicate label episode_key')
    rows = []
    for config in CONFIGS:
        counts = []
        for run in range(1, 6):
            root = RESULTS / 'report_inputs/ablation_study/rerun_20260821'
            source_config = config
            if config == 'qwen_full':
                root = root / 'qwen3.5-9b'
                source_config = 'full'
            suffix = f'{source_config}/run_{run}'
            summary = json.loads((root / f'validation/{suffix}/summary.json').read_text(encoding='utf-8'))
            predictions = replay_predictions(root / f'logs/{suffix}/02_lms_log_stream.log')
            if predictions.keys() != keys:
                raise ValueError(f'{config}/{run}: episode keys differ from labels')
            macro = {category: group for group, categories in summary['macro_groups'].items()
                     for category in categories}
            fine, coarse = Counter(), Counter()
            by_key = {}
            for label in labels:
                key = label['episode_key']
                prediction = predictions[key]
                category = prediction['root_cause_category'].strip().lower()
                group = macro.get(category, 'unknown')
                evidence = prediction.get('evidence') or []
                evidence = [evidence] if isinstance(evidence, str) else [str(x) for x in evidence[:6]]
                row = dict(zip(FIELDS, [config, str(run), key, category,
                           str(prediction.get('confidence', 0.0)),
                           json.dumps(evidence, ensure_ascii=False), group]))
                rows.append(row)
                by_key[key] = row
                fine[label['ground_truth_category'], category] += 1
                coarse[label['ground_truth_macro'], group] += 1
            for counter, metric_key in [(fine, 'ground_truth_vs_llm_metrics'),
                                        (coarse, 'ground_truth_vs_llm_macro')]:
                metrics = summary[metric_key]
                expected = Counter({(a, b): n for a, values in metrics['confusion_matrix'].items()
                                    for b, n in values.items() if n})
                correct = sum(n for (a, b), n in counter.items() if a == b)
                if (counter != expected or metrics['total'] != len(labels)
                        or abs(correct / len(labels) - metrics['accuracy']) > 1e-12):
                    raise ValueError(f'{config}/{run}: {metric_key} differs from archived summary')
            for mismatch in summary['ground_truth_vs_llm_mismatches']:
                for field in FIELDS[3:]:
                    value = by_key[mismatch['episode_key']][field]
                    expected = mismatch[field]
                    if field == 'llm_evidence':
                        value = json.loads(value)
                    elif field == 'llm_confidence':
                        value = float(value)
                    if value != expected:
                        raise ValueError(f'{config}/{run}: mismatch row differs: {field}')
            counts.append(sum(n for (a, b), n in fine.items() if a == b))
        print(f'{config}: {counts}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = build()  # Validate all sources before writing anything.
    path = RESULTS / 'predictions.csv'
    if args.check:
        with path.open(encoding='utf-8', newline='') as handle:
            existing = list(csv.DictReader(handle))
        expected = {(r['configuration'], r['run'], r['episode_key']): r for r in rows}
        actual = {(r['configuration'], r['run'], r['episode_key']): r for r in existing}
        if len(existing) != len(rows) or actual != expected:
            raise SystemExit('predictions.csv differs from the archived replay predictions')
        print(f'Verified {len(rows)} predictions against all 30 archived summaries.')
    else:
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        path.write_bytes(buffer.getvalue().encode('utf-8'))
        print(f'Wrote {len(rows)} predictions to {path}')


if __name__ == '__main__':
    main()
