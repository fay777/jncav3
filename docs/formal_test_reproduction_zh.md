# JNCA 正式实验测试与复现流程

## 1. 实验角色与数据隔离

正式实验把训练、验证和测试严格分离：

- 训练集：每个训练随机种子独立生成，仅用于参数更新；
- 冻结验证集：`benchmark_jnca_v1/validation/seed_10001`，仅用于选择 checkpoint；
- 冻结主测试集：`benchmark_jnca_v1/test/seed_20001` 至 `seed_20005`，只在 checkpoint 选定后执行推理；
- 冻结物理网络：`benchmark_jnca_v1/p_net`，所有方法共用。

测试命令必须使用 `--regenerate_v_nets false`。测试过程中不得更新模型参数，也不得根据测试结果重新选择 checkpoint 或调整指标阈值。

## 2. Checkpoint 选择规则

每个学习方法分别训练 5 个随机种子。训练期间每隔 10 个 epoch 在冻结验证集上评估。每个训练种子按以下预先确定的规则选择一个 checkpoint：

1. 最大化验证集 `long_term_aver_time_revenue`；
2. 若第一指标相同，则选择 `time_r2c_ratio` 更高者。

冻结测试脚本只加载已由验证集选出的 `best_model.pkl`。测试集不参与 checkpoint 选择。

## 3. 保存 checkpoint 审计清单

`scripts/run_frozen_learning_tests.sh` 会在推理开始前创建：

```text
<测试结果目录>/checkpoint_manifest.tsv
```

清单对每个方法和训练种子记录：模型类型、checkpoint 绝对路径、SHA-256 和文件大小。测试目录内新建的 `model/` 为空是正常现象：正式测试使用 `--num_train_epochs 0`，只读取原训练目录中的权重，不会保存新模型。

对于已经完成的测试，可在不重跑推理的情况下补写清单：

```bash
MAIN_REPO=/home/zhangpeiying01/lfy/JNCA_repro_v3_paper_consistent
TRUST_REPO=/home/zhangpeiying01/lfy/JNCA_repro_v3_truststate
cd "${MAIN_REPO}"

env \
  JNCA_CHECKPOINT_MANIFEST_PATH="${MAIN_REPO}/runs_formal_20261026_learning_security_test/checkpoint_manifest.tsv" \
  JNCA_EGAT_NO_TRUST_ROOT="${TRUST_REPO}/runs_formal_20261023_fixedmix_corrected" \
  JNCA_EGAT_TRUST_ROOT="${TRUST_REPO}/runs_formal_20261023_fixedmix_truststate_corrected" \
  JNCA_A2C_ROOT="${MAIN_REPO}/runs_formal_20261022/a2c_gcn" \
  JNCA_TWO_STAGE_ROOT="${MAIN_REPO}/runs_formal_20261022/two_stage_gcn_drl_adapted" \
  bash scripts/write_checkpoint_manifest.sh
```

应得到 20 条数据记录，即 4 个学习方法乘以 5 个训练种子。可用以下命令复核：

```bash
wc -l runs_formal_20261026_learning_security_test/checkpoint_manifest.tsv
```

包含表头时预期为 `21`。

## 4. 冻结测试运行

在代码版本、冻结数据和 checkpoint 路径均确认后运行：

```bash
MAIN_REPO=/home/zhangpeiying01/lfy/JNCA_repro_v3_paper_consistent
TRUST_REPO=/home/zhangpeiying01/lfy/JNCA_repro_v3_truststate
cd "${MAIN_REPO}"

nohup env \
  JNCA_BENCHMARK_ROOT="${MAIN_REPO}/benchmark_jnca_v1" \
  JNCA_TEST_RUN_ROOT="${MAIN_REPO}/runs_formal_YYYYMMDD_learning_security_test" \
  JNCA_TEST_GPUS="2 3 4 5 6" \
  JNCA_EGAT_NO_TRUST_ROOT="${TRUST_REPO}/runs_formal_20261023_fixedmix_corrected" \
  JNCA_EGAT_TRUST_ROOT="${TRUST_REPO}/runs_formal_20261023_fixedmix_truststate_corrected" \
  JNCA_A2C_ROOT="${MAIN_REPO}/runs_formal_20261022/a2c_gcn" \
  JNCA_TWO_STAGE_ROOT="${MAIN_REPO}/runs_formal_20261022/two_stage_gcn_drl_adapted" \
  bash scripts/run_frozen_learning_tests.sh \
  > "${MAIN_REPO}/runs_formal_YYYYMMDD_learning_security_test_launcher.log" 2>&1 &
```

脚本拒绝覆盖已存在的结果目录，避免新旧结果混合。

## 5. 完整性检查

学习方法应产生 100 个汇总文件：

```text
4 个方法 × 5 个训练种子 × 5 个冻结测试集 = 100
```

基线应产生 25 个汇总文件：

```text
5 个基线 × 5 个冻结测试集 = 25
```

检查命令：

```bash
find runs_formal_YYYYMMDD_learning_security_test -name global_summary.csv -type f | wc -l
find runs_formal_YYYYMMDD_baselines_security_test -name global_summary.csv -type f | wc -l
grep -R -E 'Traceback|Load failed|RuntimeError|FileNotFoundError|KeyError' \
  runs_formal_YYYYMMDD_learning_security_test/logs \
  runs_formal_YYYYMMDD_baselines_security_test/logs
```

学习方法日志中，每个训练种子应出现 5 次 `Loaded pretrained model from`，并且不应出现 `Load failed`。

## 6. 主指标与统计方法

论文主实验报告：

- 总接纳率；
- 安全敏感请求接纳率；
- 静态安全合规率；
- `LCB-0.7`、`LCB-0.8` 与 `LCB-0.9` 条件满足率；
- `LCB-0.7`、`LCB-0.8` 与 `LCB-0.9` 可靠安全部署率；
- `mean_min_node_reliability_lcb`；
- `LTA Time Revenue`；
- `Time R2C`。

旧耦合式 `srsr` 仅为兼容历史结果保留，不作为主结论。

学习方法不能把 25 次测试当作 25 个独立训练样本。正确汇总顺序为：

1. 对每个训练种子的 5 个冻结测试集求平均；
2. 对 5 个训练种子的均值报告总体均值和样本标准差。

无训练过程的基线直接在 5 个冻结测试集上报告均值和样本标准差。

## 7. 投稿归档清单

正式归档至少保留：

- Git commit SHA；
- Python、PyTorch、PyG 和 CUDA 版本；
- 冻结物理网络与请求集的 SHA-256；
- `checkpoint_manifest.tsv`；
- 每次测试的 `config.yaml`；
- `global_summary.csv` 和逐请求 `records/*.csv`；
- 完整运行日志；
- 最终论文汇总表及生成脚本。

绝对路径只用于记录当时的服务器位置；checkpoint 身份以 SHA-256 为准。
