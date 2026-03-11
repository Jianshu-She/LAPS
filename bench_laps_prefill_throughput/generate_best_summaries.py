"""
Generate "best result" summaries for all multi-run benchmark directories.
For each setting/cc, picks the best value across runs:
  - throughput metrics: MAX
  - latency metrics: MIN
  - duration: MIN
"""

import json, os, glob

BASE = os.path.dirname(os.path.abspath(__file__))

# All result dirs that have run subdirectories
RESULT_DIRS = sorted(glob.glob(os.path.join(BASE, "results_*")))

# Higher is better
HIGHER_BETTER = {'prefill_throughput_tok_s', 'request_throughput_req_s'}
# Lower is better
LOWER_BETTER = {'median_ttft_ms', 'mean_ttft_ms', 'p50_ttft_ms', 'p90_ttft_ms', 'p99_ttft_ms', 'total_duration_s'}

METRICS = [
    ('prefill_throughput_tok_s', 'PREFILL THROUGHPUT (tokens/s)',  '{:>8.0f}'),
    ('request_throughput_req_s', 'REQUEST THROUGHPUT (req/s)',     '{:>8.1f}'),
    ('median_ttft_ms',          'MEDIAN TTFT (ms)',                '{:>8.1f}'),
    ('mean_ttft_ms',            'MEAN TTFT (ms)',                  '{:>8.1f}'),
    ('p50_ttft_ms',             'P50 TTFT (ms)',                   '{:>8.1f}'),
    ('p90_ttft_ms',             'P90 TTFT (ms)',                   '{:>8.1f}'),
    ('p99_ttft_ms',             'P99 TTFT (ms)',                   '{:>8.1f}'),
    ('total_duration_s',        'TOTAL DURATION (s)',              '{:>8.1f}'),
]

# Auto-detect settings and cc levels from json files
def detect_config(run_dirs):
    """Scan json files to find all settings and concurrency levels."""
    settings = set()
    ccs = set()
    for rd in run_dirs:
        for f in glob.glob(os.path.join(rd, "*.json")):
            name = os.path.basename(f).replace('.json', '')
            # pattern: {setting}_cc{num}
            parts = name.rsplit('_cc', 1)
            if len(parts) == 2:
                settings.add(parts[0])
                try:
                    ccs.add(int(parts[1]))
                except ValueError:
                    pass
    return sorted(settings), sorted(ccs)


def make_labels(settings):
    """Generate display labels from setting names."""
    label_map = {
        'vanilla_sglang': 'Vanilla SGLang',
        'prefill_cuda_graph': 'Prefill CUDA Graph',
        'batch_prefill_cuda_graph': 'Batch Prefill CUDA Graph',
        'prefill_disagg': 'Prefill Disaggregation',
        'laps': 'LAPS',
        'laps_n1': 'LAPS (N=1)',
        'laps_n2': 'LAPS (N=2)',
        'laps_n4': 'LAPS (N=4)',
    }
    return {s: label_map.get(s, s) for s in settings}


def process_dir(results_dir):
    """Process one results directory."""
    # Find run subdirectories
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "*_run*")))
    if not run_dirs:
        return None

    num_runs = len(run_dirs)
    settings, ccs = detect_config(run_dirs)
    if not settings or not ccs:
        return None

    labels = make_labels(settings)
    dir_name = os.path.basename(results_dir)

    # Load all data
    all_data = {}
    for ri, rd in enumerate(run_dirs, 1):
        for s in settings:
            for cc in ccs:
                f = os.path.join(rd, f'{s}_cc{cc}.json')
                if os.path.exists(f):
                    with open(f) as fh:
                        all_data[(s, cc, ri)] = json.load(fh)

    # Compute best
    col_width = max(15, max(len(f"cc={cc}") for cc in ccs) + 2)
    lines = []
    lines.append('')
    lines.append('=' * 170)
    lines.append(f'  BEST RESULT SUMMARY — {dir_name} ({num_runs} runs)')
    lines.append('=' * 170)

    for key, section_title, fmt in METRICS:
        lines.append('')
        lines.append(f'  {section_title}')
        lines.append('-' * 170)
        hdr = f"{'Setting':<28s}"
        for cc in ccs:
            hdr += f'  {"cc="+str(cc):>{col_width}s}'
        lines.append(hdr)
        lines.append('-' * 170)

        for s in settings:
            row = f'{labels[s]:<28s}'
            for cc in ccs:
                vals = []
                for ri in range(1, num_runs + 1):
                    d = all_data.get((s, cc, ri))
                    if d and key in d:
                        vals.append(d[key])
                if vals:
                    if key in HIGHER_BETTER:
                        best = max(vals)
                    else:
                        best = min(vals)
                    row += '  ' + fmt.format(best).rjust(col_width)
                else:
                    row += '  ' + 'N/A'.rjust(col_width)
            lines.append(row)

    lines.append('')
    lines.append('=' * 170)

    output = '\n'.join(lines)

    # Save
    out_path = os.path.join(results_dir, 'best_summary.txt')
    with open(out_path, 'w') as f:
        f.write(output + '\n')

    return out_path, output


# Main
generated = []
for rd in RESULT_DIRS:
    result = process_dir(rd)
    if result:
        out_path, output = result
        print(output)
        print(f'\nSaved to {out_path}')
        print()
        generated.append(out_path)

if generated:
    print(f'\n=== Generated {len(generated)} best-result summaries ===')
    for p in generated:
        print(f'  {p}')
else:
    print('No multi-run result directories found.')
