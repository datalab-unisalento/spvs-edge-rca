"""Rebuild the selected reports from existing rerun summaries and logs.

No LLM requests, relabeling, or per-episode validation are performed. The
original report generators and their aggregation helpers are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import run_ablation as ra
import run_ablation_repeats as rr

RESULTS = Path(__file__).resolve().parents[1]
PACKAGE = RESULTS.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=RESULTS/'report_inputs',
                        help='Packaged input directory; must be inside cameraready/.')
    parser.add_argument('--output', type=Path, default=RESULTS/'reproduced',
                        help='Output directory; must be inside cameraready/results/.')
    parser.add_argument('--check-inputs', action='store_true',
                        help='Check required files only; do not calculate or write reports.')
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    if PACKAGE not in source.parents:
        parser.error('--source-root must be a subdirectory of cameraready/')
    if RESULTS not in output.parents:
        parser.error('--output must be a subdirectory of cameraready/results/')
    gemma = source/'ablation_study/rerun_20260821'
    qwen = gemma/'qwen3.5-9b'
    native_a = gemma/'lms_stream_gemma.jsonl'
    native_b = qwen/'lms_stream_qwen.jsonl'
    required = [native_a, native_b]
    for directory, configs in [(gemma, ra.CONFIGS), (qwen, ['full'])]:
        for config in configs:
            for run in range(1, 6):
                required.extend([
                    directory/f'validation/{config}/run_{run}/summary.json',
                    directory/f'logs/{config}/run_{run}/02_lms_log_stream.log',
                ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit('Missing report inputs:\n'+'\n'.join(missing))
    if any(PACKAGE not in path.resolve().parents for path in required):
        parser.error('Report inputs must resolve inside cameraready/ (including symlink targets)')
    manifest_path = source/'MANIFEST.json'
    if PACKAGE not in manifest_path.resolve().parents:
        parser.error('Input manifest must resolve inside cameraready/')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {str(path.relative_to(source).as_posix()): path for path in required}
    required_manifest = [entry for entry in manifest if entry['path'] in expected]
    if {entry['path'] for entry in required_manifest} != set(expected) or len(required_manifest) != len(required):
        raise SystemExit('Input manifest does not match the required report inputs')
    for entry in required_manifest:
        path = expected[entry['path']]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise SystemExit(f'Input hash mismatch: {path}')
    print(f'Found all {len(required)} required report inputs (30 summaries, 30 trial logs, 2 native JSONL streams).')
    print('All packaged input SHA-256 hashes match.')
    if args.check_inputs:
        return

    output.mkdir(parents=True, exist_ok=True)
    ablation = {}
    for config in ra.CONFIGS:
        trials = []
        for run in range(1, 6):
            summary = json.loads((gemma/f'validation/{config}/run_{run}/summary.json').read_text(encoding='utf-8'))
            log = gemma/f'logs/{config}/run_{run}/02_lms_log_stream.log'
            summary['prompt_length'] = ra.compute_prompt_length_stats(log)
            summary['failed_calls'] = ra.count_failed_calls(log)
            # Preserve the status recorded in the historical report; no new inference is made.
            trials.append((run, 'inferred', summary))
        ablation[config] = rr.aggregate_config(config, trials)
    rr.write_comparison(output, ablation)

    scripts = Path(__file__).resolve().parent
    commands = [
        [sys.executable, '-B', str(scripts/'compare_models.py'),
         '--model-a-dir', str(gemma), '--model-b-dir', str(qwen),
         '--model-a-label', 'gemma-4-e4b', '--model-b-label', 'qwen3.5-9b-q4_k_m',
         '--model-a-stats-log', str(native_a),
         '--model-b-stats-log', str(native_b),
         '--out', str(output/'model_comparison_rerun')],
        [sys.executable, '-B', str(scripts/'macro_confusion_matrix.py'),
         '--model', f'gemma-4-e4b={gemma}', '--model', f'qwen3.5-9b={qwen}',
         '--repeats', '5', '--out', str(output/'macro_confusion_matrix')],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    for name in ['comparison.md', 'model_comparison_rerun.md', 'macro_confusion_matrix.md']:
        matches = (output/name).read_bytes() == (RESULTS/name).read_bytes()
        print(f'{name}: '+('matches the supplied report byte-for-byte' if matches else 'differs from the supplied report; compare the files before drawing conclusions'))


if __name__ == '__main__':
    main()
