"""Generate compact summaries for completed formal experiments."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


TEST_METRICS = (
    'acceptance_rate',
    'security_sensitive_acceptance_rate',
    'r2c_ratio',
    'time_r2c_ratio',
    'long_term_aver_time_revenue',
    'static_security_satisfaction_rate',
    'lcb_0_7_rate',
    'lcb_0_8_rate',
    'mean_min_node_reliability_lcb',
    'reliable_secure_deployment_rate_lcb_0_7',
    'reliable_secure_deployment_rate_lcb_0_8',
)


def read_single_row(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return next(csv.DictReader(handle))


def optional_float(row, key):
    value = row.get(key, '')
    return float(value) if value not in ('', None) else None


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return '', ''
    return f'{mean(values):.6f}', f'{stdev(values):.6f}' if len(values) > 1 else '0.000000'


def aggregate_test(method, files, source_group):
    records = [read_single_row(path) for path in sorted(files)]
    output = {'method': method, 'source_group': source_group, 'n_test_seeds': len(records)}
    for metric in TEST_METRICS:
        output[f'{metric}_mean'], output[f'{metric}_sd'] = describe(
            [optional_float(record, metric) for record in records]
        )
    return output


def aggregate_validation(method, files):
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(files)]
    output = {'method': method, 'n_train_seeds': len(records)}
    for key in ('model_epoch', 'long_term_aver_time_revenue', 'time_r2c_ratio'):
        output[f'{key}_mean'], output[f'{key}_sd'] = describe(
            [float(record[key]) for record in records]
        )
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_files(method, directory, pattern, expected):
    files = list(directory.rglob(pattern))
    if len(files) != expected:
        raise RuntimeError(
            f'{method}: expected {expected} files matching {pattern}, '
            f'found {len(files)} in {directory}'
        )
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/formal_20261025'))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()

    baseline_sources = {
        'First-Fit': root / 'runs_formal_20261022/baselines_cpu/first_fit',
        'RW-RANK-BFS': root / 'runs_formal_20261022/baselines_cpu/rw_rank_bfs',
        'MCTS': root / 'runs_formal_20261022/baselines_cpu/mcts',
        'APPF-DSI-adapted': root / 'runs_formal_20261025_security_baselines_fixed/appf_dsi_adapted',
        'SSD-OSE-adapted': root / 'runs_formal_20261025_security_baselines_fixed/ssd_ose_adapted',
    }
    baseline_rows = []
    for method, directory in baseline_sources.items():
        files = require_files(method, directory, 'global_summary.csv', 5)
        baseline_rows.append(aggregate_test(method, files, str(directory.relative_to(root))))
    write_csv(output_dir / 'baseline_test_summary.csv', baseline_rows)

    validation_sources = {
        'EGAT-DPOT (no trust state)': root / 'runs_formal_20261023_fixedmix_corrected',
        'EGAT-DPOT (+ trust state)': root / 'runs_formal_20261023_fixedmix_truststate_corrected',
        'A2C-GCN': root / 'runs_formal_20261022/a2c_gcn',
        'TwoStage-GCN-DRL-adapted': root / 'runs_formal_20261022/two_stage_gcn_drl_adapted',
    }
    validation_rows = []
    for method, directory in validation_sources.items():
        files = require_files(method, directory, 'best_checkpoint.json', 5)
        validation_rows.append(aggregate_validation(method, files))
    write_csv(output_dir / 'learning_validation_summary.csv', validation_rows)

    (output_dir / 'README.md').write_text(
        '# 正式实验结果索引\n\n'
        '- `baseline_test_summary.csv`：五个 baseline 在冻结测试集 `seed_20001--20005` 上的均值和样本标准差。\n'
        '- `learning_validation_summary.csv`：四组学习方法的验证集 checkpoint 选择结果，不是最终测试结果。\n'
        '- First-Fit、RW-RANK-BFS、MCTS 的原始结果位于 `runs_formal_20261022/baselines_cpu/`。\n'
        '- APPF-DSI-adapted、SSD-OSE-adapted 的修复版原始结果位于 `runs_formal_20261025_security_baselines_fixed/`。\n'
        '- 安全指标列为空表示源实验早于新统计模块。LCB 无法从旧汇总可靠重建，必须加载最佳 checkpoint 后在冻结测试集重新推理。\n'
        '- 历史 `srsr` 不进入本索引和论文主表；正式安全结果使用静态安全合规率、敏感请求接纳率、LCB-0.7/0.8 和可靠安全部署率。\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
