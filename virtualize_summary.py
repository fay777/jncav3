import pandas as pd
import matplotlib.pyplot as plt
import os

from utils.virtualize import virtualize_model_metrics, Visualization_epoch, process_csv_to_metrics

# # 使用示例
# csv_file = 'save/gat_seq2seq_ppo/LAPTOP-M43DF0FU-20250521T144535/training_metrics.csv'  # 替换为你的CSV文件路径
# metrics = process_csv_to_metrics(csv_file)
# Visualization_epoch(metrics)



# 设置文件路径（假设你的5个csv文件名为 model_1.csv 到 model_5.csv）
file_paths = [
        "save/first_fit/LAPTOP-M43DF0FU-20250520T172851/run_metrics.csv",
        "save/rw_rank_bfs/LAPTOP-M43DF0FU-20250520T222322/run_metrics.csv",
        "save/a3c_gcn_seq2seq/LAPTOP-M43DF0FU-20250522T132823/run_metrics.csv",
        "save/gat_seq2seq_ppo/LAPTOP-M43DF0FU-20250522T124225/run_metrics.csv"
    ]

# 定义模型名称（与文件路径顺序对应）
model_names = ["first_fit",
                "rw_rank_bfs",
                "a3c_gcn_seq2seq",
                "gat_seq2seq_ppo"
                ]
virtualize_model_metrics(file_paths,model_names)