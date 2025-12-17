import os
import csv
from statistics import mean
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 满分要求：保持目标分数线
AVERAGE_SCORE_TO_SOLVE = 475
CONSECUTIVE_RUNS_TO_SOLVE = 100

class ScoreLogger:
    """记录每个算法的得分并生成独立的图表，支持动态命名。"""

    def __init__(self, env_name: str, log_name: str = "run"):
        self.env_name = env_name
        self.log_name = log_name
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)

        # 动态定义保存目录和文件路径
        self.target_dir = "./scores"
        os.makedirs(self.target_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.target_dir, f"{log_name}.csv")
        self.png_path = os.path.join(self.target_dir, f"{log_name}.png")

        # 如果存在旧的实验记录，则删除（防止数据混淆）
        if os.path.exists(self.csv_path): os.remove(self.csv_path)
        if os.path.exists(self.png_path): os.remove(self.png_path)

    def add_score(self, score: float, run: int):
        """添加分数并更新该算法专属的图表。"""
        self._save_csv(self.csv_path, score)
        self._save_png(
            input_path=self.csv_path,
            output_path=self.png_path,
            x_label="Episodes",
            y_label="Scores",
            average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
            show_goal=True,
            show_trend=True, 
            show_legend=True,
        )

        self.scores.append(score)
        mean_score = mean(self.scores)
        print(f"[{self.log_name}] Ep: {run}, Last: {score}, Avg: {mean_score:.2f}")

    def _save_png(self, input_path, output_path, x_label, y_label,
                  average_of_n_last, show_goal, show_trend, show_legend):
        """完整保留原始绘图逻辑：包含 Goal Line, Trend Line, MA。"""
        if not os.path.exists(input_path):
            return

        x, y = [], []
        with open(input_path, "r") as scores_file:
            reader = csv.reader(scores_file)
            for i, row in enumerate(reader):
                x.append(i)
                y.append(float(row[0]))

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label="Score per Episode", alpha=0.5)

        # 1. 最近 100 步平均值 (红色虚线)
        if average_of_n_last is not None and len(x) > 0:
            avg_range = min(average_of_n_last, len(x))
            plt.plot(x[-avg_range:], [np.mean(y[-avg_range:])] * avg_range,
                     linestyle="--", color="orange", label=f"Avg Last {avg_range}")

        # 2. 目标线 475 (绿色点线)
        if show_goal:
            plt.axhline(y=AVERAGE_SCORE_TO_SOLVE, color='green', linestyle=":", 
                        label=f"Goal ({AVERAGE_SCORE_TO_SOLVE})")

        # 3. 趋势线 (橙色点划线)
        if show_trend and len(x) > 1:
            y_trend = []
            block = []
            for val in y:
                block.append(val)
                y_trend.append(np.mean(block))
                if len(block) >= 100: block = [] # 每100步重置一次均值计算
            plt.plot(x, y_trend, linestyle="-.", color="red", label="Trend")

        plt.title(f"{self.env_name} Training: {self.log_name}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if show_legend: plt.legend(loc="upper left")
        plt.grid(True, alpha=0.2)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score: float):
        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([score])