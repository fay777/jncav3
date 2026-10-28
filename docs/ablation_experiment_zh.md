# EGAT-DPOT 消融实验说明

## 实验矩阵

完整模型配置为：归一化安全得分、增量式安全 shaping、固定安全权重 0.5、trust observation、失败惩罚 1.0。

| 版本 | 归一化 TS | 增量 shaping | 安全权重 | trust state | 用途 |
| --- | --- | --- | --- | --- | --- |
| Full | 是 | 是 | 0.5 | 是 | 完整方法，已有正式结果 |
| No trust state | 是 | 是 | 0.5 | 否 | trust observation 消融，已有正式结果 |
| No incremental shaping | 是 | 否 | 0.5 | 是 | 检验增量奖励的作用 |
| Adaptive security weight | 是 | 是 | SSI 自适应 | 是 | 检验固定 0.5/0.5 权重 |
| No security reward | 是 | 是 | 0.0 | 是 | 检验安全奖励的贡献 |
| Legacy reward | 否 | 否 | 0.5 | 是 | 原始未归一化、累积奖励整体回退；属于组合消融 |

`Legacy reward` 同时关闭归一化和增量 shaping，因为代码明确要求增量 shaping 必须基于归一化安全得分。该版本不能解释为单独的“无归一化”消融，应在论文中标为旧奖励设计整体回退。

## 训练与测试

新增版本均训练 5 个随机种子，共用冻结物理网络和冻结验证集 `seed_10001`。每 10 个 epoch 验证一次，按验证集 LTA Time Revenue 最大、Time R2C 次优的规则保存 `best_model.pkl`。

测试阶段只加载验证集选出的 checkpoint，在冻结测试请求 `seed_20001` 至 `seed_20005` 上推理，不更新模型参数。

学习方法的统计单位是训练种子：先对同一个训练种子的 5 个测试集求平均，再跨 5 个训练种子报告均值和样本标准差。

## 论文建议指标

消融表至少报告：总接纳率、安全敏感请求接纳率、LCB-0.7/0.8、可靠安全部署率 0.7/0.8、LTA Time Revenue、Time R2C。静态安全合规率作为实现正确性检查，应保持 100%。旧耦合式 SRSR 不进入消融主结论。
