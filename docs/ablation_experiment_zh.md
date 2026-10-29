# EGAT-DPOT 消融实验说明

## 核心组件消融

完整模型配置为：归一化安全得分、增量式安全 shaping、固定安全权重 0.5、trust observation、失败惩罚 1.0。

| 版本 | 归一化 TS | 安全奖励结算 | 安全权重 | trust state | 用途 |
| --- | --- | --- | --- | --- | --- |
| Full | 是 | 增量式 | 0.5 | 是 | 完整方法，已有正式结果 |
| No trust state | 是 | 增量式 | 0.5 | 否 | 删除历史可靠性观测，已有正式结果 |
| No incremental shaping | 是 | SFC 完成时一次结算 | 0.5 | 是 | 只删除稠密的增量安全反馈，保持最终目标不变 |
| No security reward | 是 | 增量式 | 0.0 | 是 | 仅优化效率，检验安全奖励的贡献 |
| No efficiency reward | 是 | 增量式 | 1.0 | 是 | 仅优化安全，检验效率奖励对平衡的贡献 |
| EGAT to GCN | 是 | 增量式 | 0.5 | 是 | 用同为两层、同输出维度的拓扑 GCN 替换边属性 EGAT |
| Without LSTM | 是 | 增量式 | 0.5 | 是 | 用逐位置 MLP 与均值池化替换序列 LSTM 编码器和解码器 |

`No incremental shaping` 不是恢复旧版逐 VNF 累计奖励。中间成功步骤奖励为 0，完整 SFC 成功时才结算一次
`0.5 * TS_norm + 0.5 * Revenue/Cost`；失败时使用固定终止惩罚。这样与 Full 的成功回报总目标一致，唯一变化是是否提供逐步安全反馈。

`No security reward`、Full 和 `No efficiency reward` 分别对应安全权重 0、0.5 和 1，可直接展示“只优化效率—联合优化—只优化安全”的安全与效率权衡。

## 补充敏感性与旧设计对照

| 版本 | 性质 | 论文位置建议 |
| --- | --- | --- |
| Adaptive security weight | 固定 0.5 改为 SSI 自适应权重 | 权重策略敏感性分析 |
| Legacy reward | 同时关闭 TS 归一化与增量 shaping | 附录中的旧奖励整体回退 |

`Legacy reward` 同时关闭归一化和增量 shaping，因为代码明确要求增量 shaping 必须基于归一化安全得分。该版本不能解释为单独的“无归一化”消融，应在论文中标为旧奖励设计整体回退。

训练与测试脚本默认只运行三个尚缺结果的核心版本：

```text
no_incremental_shaping no_security_reward no_efficiency_reward
```

网络结构消融单独使用新的结果目录运行：

```bash
JNCA_ABLATION_VARIANTS="egat_to_gcn without_lstm"
```

`EGAT to GCN` 保留两层图编码器和 64 维输出，但取消多头注意力与链路属性条件化，只使用物理拓扑邻接关系。`Without LSTM` 保留相同输入、输出维度、注意力接口、PPO、reward、mask 和 trust state；VNF 表征由逐位置 MLP 生成，均值池化提供初始状态，解码阶段使用前馈变换，不再使用循环记忆。两个版本都必须从头训练，不能用 A2C-GCN 或 TwoStage-GCN-DRL 的结果替代。

若需要补充实验，显式设置：

```bash
JNCA_ABLATION_VARIANTS="adaptive_security_weight legacy_reward"
```

## 训练与测试

新增核心版本均训练 5 个随机种子，共用冻结物理网络和冻结验证集 `seed_10001`。每 10 个 epoch 验证一次，按验证集 LTA Time Revenue 最大、Time R2C 次优的规则保存 `best_model.pkl`。

默认核心任务应产生 `3 × 5 = 15` 个 `best_model.pkl`。Full 和 No trust state 直接复用已经完成的正式训练结果，不重复训练。

结构消融另外产生 `2 × 5 = 10` 个 `best_model.pkl`。

测试阶段只加载验证集选出的 checkpoint，在冻结测试请求 `seed_20001` 至 `seed_20005` 上推理，不更新模型参数。

默认核心冻结测试应产生 `3 × 5 × 5 = 75` 个 `global_summary.csv`。与既有 Full、No trust state 合并后，核心消融表共 5 个版本。

结构消融冻结测试另外产生 `2 × 5 × 5 = 50` 个 `global_summary.csv`。合并奖励、状态和结构消融后，主消融表共 7 个版本。

学习方法的统计单位是训练种子：先对同一个训练种子的 5 个测试集求平均，再跨 5 个训练种子报告均值和样本标准差。

## 论文建议指标

核心消融表至少报告：总接纳率、安全敏感请求接纳率、LCB-0.7/0.8/0.9、可靠安全部署率 0.7/0.8/0.9、LTA Time Revenue、Time R2C。静态安全合规率作为实现正确性检查，应保持 100%。旧耦合式 SRSR 不进入消融主结论。

推荐按以下关系解释：

- Full 对比 No trust state：历史可靠性状态的贡献；
- Full 对比 No incremental shaping：增量式安全反馈的贡献；
- No security reward、Full、No efficiency reward：安全—效率平衡；
- Full 对比 EGAT to GCN：边属性增强注意力图编码器的贡献；
- Full 对比 Without LSTM：VNF 顺序与历史部署序列建模的贡献；
- Adaptive security weight 与 Legacy reward：只作为补充分析，不用于单组件因果归因。
