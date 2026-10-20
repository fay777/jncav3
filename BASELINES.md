# Comparable baselines

All solvers below use the same `SolutionStepEnvironment`, `Controller`,
`Counter`, and `Recorder` as `gat_seq2seq_ppo`.  Consequently CPU/memory,
static node security, bandwidth, and static link-security feasibility are
checked by the same code path for every method.

| CLI solver name | Paper label | Scope |
| --- | --- | --- |
| `first_fit` | First-Fit | Greedy heuristic |
| `rw_rank_bfs` | RW-RANK-BFS | Cheng et al. (2011) inspired ranking + BFS placement |
| `mcts` | MCTS | Haeri et al. (2018) inspired search baseline |
| `a2c_gcn_seq2seq` | A2C-GCN | Accurately named legacy synchronous GCN--Seq2Seq implementation |
| `two_stage_gcn_a2c_adapted` | TwoStage-GCN-DRL-adapted | Chen et al. (2023) inspired node-placement then constrained-routing design |
| `appf_dsi_adapted` | APPF-DSI-adapted | Wang et al. (2024) inspired security-capability greedy design |
| `ssd_ose_adapted` | SSD-OSE-adapted | Wang et al. (2025) inspired security-cost/overflow-elimination design |

`APPF-DSI-adapted` and `SSD-OSE-adapted` are **not direct reproductions**:
this repository has fixed VNF types and no multi-vendor, multi-version SF, or
backup-chain state.  `TwoStage-GCN-DRL-adapted` is likewise not a direct
reproduction because the common setting has one substrate rather than multiple
datacenters.  These names and limitations must be retained in the manuscript.

For every final comparison, keep the physical-network dataset fixed and run the
same request workload seed once per method.  Run learning baselines for the
same number of training epochs as the proposed PPO model; evaluate heuristic
baselines without training.  The request-level record CSVs are required for
any post-hoc LCB analysis.
