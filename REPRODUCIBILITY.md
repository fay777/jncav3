# JNCA 论文复现实现说明

本目录是依据 JNCA 稿件整理的、面向可复现实验的实现版本。原始项目未被修改；其冻结位置和说明见
[`FROZEN_SOURCE.md`](FROZEN_SOURCE.md)。

## 已落实的论文设置

- Waxman 物理拓扑：`alpha=0.4`、`beta=0.4`。
- 虚拟 SFC 生命周期：在 `[50, 200]` 上进行离散均匀采样。
- 每条 SFC 均显式包含标记为 `normal` 和 `secure` 的两类 VNF；每一类的数量独立地在 `[1, 5]` 内采样。安全 VNF 使用 `settings/v_sim_setting.yaml` 中配置的更高安全需求范围。
- 策略网络：两层使用边特征的图注意力层、四个注意力头、64 维输出；两层 LSTM；PPO 优化。
- 必须指定随机种子。种子同时控制 Python、NumPy 和 PyTorch 的随机数生成器，并启用确定性的 CuDNN 设置。
- 环境重置与 PPO 回滚使用兼容 NetworkX 3.x 的网络复制方法，避免对 NetworkX 视图进行不安全的 `deepcopy`。

## 服务器虚拟环境

建议使用 Conda 在服务器端创建独立环境。当前代码已在 Python `3.8.3`、PyTorch `2.4.1`、PyTorch Geometric `2.6.1` 上验证；为最大程度减少差异，建议使用 Python 3.8：

```bash
conda create -n jnca_repro python=3.8 -y
conda activate jnca_repro
```

先用 `nvidia-smi` 查看服务器驱动支持的 CUDA 版本。以下示例使用 CUDA 12.1；若服务器环境不同，应将命令中的 `cu121` 和下载地址替换为与服务器 CUDA 对应的版本：

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch_geometric==2.6.1
pip install --upgrade -r requirements-server.txt
```

若环境中已经存在 SciPy 1.10.x，切勿安装旧版 NumPy 1.18。应先执行 `pip install --upgrade numpy==1.24.4`，再安装 `requirements-server.txt`；该版本满足 SciPy 1.10.x 的 `numpy>=1.19.5,<1.27` 约束。

完成后，在项目根目录执行以下检查；输出应显示 CUDA 可用及其设备名称：

```bash
python -c "import torch, torch_geometric; print(torch.__version__); print(torch_geometric.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

若 `torch.cuda.is_available()` 为 `False`，请不要启动正式训练；应先检查 NVIDIA 驱动、CUDA 对应的 PyTorch 安装包，以及是否在计算节点而非登录节点执行命令。

依赖安装完成后，可在项目根目录运行以下导入检查。它会导入主程序依赖，但不会开始训练：

```bash
python -c "import main; print('依赖导入检查通过')"
```

## 正式实验命令

请在本目录下、具有 CUDA 的主机上对每个种子分别运行：

```powershell
python main.py --solver_name gat_seq2seq_ppo --seed 1 --regenerate_v_nets true --num_train_epochs 140 --num_epochs 1 --use_cuda true --cuda_id 0
```

将 `--seed` 依次替换为 `2`、`3`、`4`、`5`，得到五组独立结果。`--regenerate_v_nets true` 很重要：它避免每个种子静默复用已保存的同一份虚拟网络请求集。结果写入：

`save/gat_seq2seq_ppo/<主机名>-<时间戳>/`

也可通过 `--run_dir <目录>` 为一次运行指定独立结果根目录；模型、训练信息、记录和摘要将全部写入该目录。

## 小样本端到端冒烟测试

下列命令只使用五条请求，不会修改正式的 YAML 配置文件：

```powershell
python main.py --solver_name gat_seq2seq_ppo --seed 1 --regenerate_v_nets true --num_train_epochs 1 --num_epochs 1 --use_cuda false --verbose false --if_adjust_v_sim_setting true --v_sim_setting_num_v_nets 5 --batch_size 8 --target_steps 8 --repeat_times 1
```

该冒烟测试已在本副本中成功完成，输出位于：

`save/gat_seq2seq_ppo/LAPTOP-94UHC9US-20261011T213814/`

它验证了数据生成、EGAT--LSTM 前向计算、PPO 更新和评估流程的连通性；其结果仅用于功能核验，不应作为论文实验结果报告。
