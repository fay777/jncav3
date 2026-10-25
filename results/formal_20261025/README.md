# 正式实验结果索引

- `baseline_test_summary.csv`：五个 baseline 在冻结测试集 `seed_20001--20005` 上的均值和样本标准差。
- `learning_validation_summary.csv`：四组学习方法的验证集 checkpoint 选择结果，不是最终测试结果。
- First-Fit、RW-RANK-BFS、MCTS 的原始结果位于 `runs_formal_20261022/baselines_cpu/`。
- APPF-DSI-adapted、SSD-OSE-adapted 的修复版原始结果位于 `runs_formal_20261025_security_baselines_fixed/`。
- 安全指标列为空表示源实验早于新统计模块。LCB 无法从旧汇总可靠重建，必须加载最佳 checkpoint 后在冻结测试集重新推理。
- 历史 `srsr` 不进入本索引和论文主表；正式安全结果使用静态安全合规率、敏感请求接纳率、LCB-0.7/0.8 和可靠安全部署率。
