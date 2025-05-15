import json
import os
from typing import Dict, Literal, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PERF_BASE_DIR = "runs/perf"
VIS_OUTPUT_DIR = "vis/perf"


class LatencyMetrics(TypedDict):
    e2e: Dict[Literal["mean_ms", "median_ms"], float]
    ttft: Dict[Literal["mean_ms", "median_ms", "p99_ms"], float]
    itl: Dict[Literal["mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms"], float]


class Metrics(TypedDict):
    backend: str
    batch_size: int
    successful_requests: int
    benchmark_duration_s: float
    total_input_tokens: int
    total_generated_tokens: int
    request_throughput: float
    input_token_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    concurrency: int
    latency: LatencyMetrics

def generate_bar_plot(data: Dict[str, float], x_label: str, y_label: str, y_unit: Optional[str]):
    def generate_bar_plot(data: Dict[str, float], x_label: str, y_label: str, y_unit: Optional[str]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        bars = ax.bar(list(data.keys()), list(data.values()), color=sns.color_palette("muted"))
        
        # Add labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{y_label}{f' ({y_unit})' if y_unit else ''}")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f"{height:.2f}",
                ha='center', va='bottom',
                fontsize=9
            )
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Add grid lines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-labels if there are many categories
        if len(data) > 5:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        return fig, ax


def load_metrics() -> Dict[str, Dict[str, Metrics]]:
    # benchmark -> model -> metrics
    all_metrics: Dict[str, Dict[str, Metrics]] = {}

    for model_dir in os.listdir(PERF_BASE_DIR):
        for benchmark_dir in os.listdir(os.path.join(PERF_BASE_DIR, model_dir)):
            if benchmark_dir not in all_metrics:
                all_metrics[benchmark_dir] = {}

            metrics_file = os.path.join(
                PERF_BASE_DIR, model_dir, benchmark_dir, "metrics.json"
            )
            if not os.path.exists(metrics_file):
                continue

            with open(metrics_file, "r") as f:
                metrics = Metrics(**json.load(f))

            all_metrics[benchmark_dir][model_dir] = metrics

    return all_metrics


def main() -> None:
    all_metrics: Dict[str, Dict[str, Metrics]] = load_metrics()

    sns.set_theme(style="whitegrid")


if __name__ == "__main__":
    main()
