import dataclasses
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Any

from matplotlib import pyplot as plt
from tabulate import tabulate

import psutil
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import pandas as pd
import pm4py


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def find_non_zero_ranges(float_list: List[float]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start_index: int = -1

    for i, value in enumerate(float_list):
        if value != 0 and start_index == -1:
            # Move start_index back to include preceding zeros
            start_index = i
            while start_index > 0 and float_list[start_index - 1] == 0:
                start_index -= 1
        elif value == 0 and start_index != -1:
            ranges.append((start_index, i - 1))
            start_index = -1

    if start_index != -1:
        ranges.append((start_index, len(float_list) - 1))

    # add five to each start index
    ranges = [(start + 5, end) for start, end in ranges]

    return ranges


@dataclasses.dataclass
class ConformanceMetrics:
    """Data class to store conformance metrics of a Petri net model"""

    fitness: float
    """Fitness of the model"""

    precision: float
    """Precision of the model"""

    generalization: float
    """Generalization of the model"""

    simplicity: float
    """Simplicity of the model"""

    @property
    def f1_score(self) -> float:
        return 0 if (self.precision + self.fitness) == 0 else 2 * (self.precision * self.fitness) / (self.precision + self.fitness)

    def __str__(self):
        return (
            f"F1: {self.f1_score:.2f}, F: {self.fitness:.2f}, P: {self.precision:.2f}, G: {self.generalization:.2f}, "
            f"S: {self.simplicity:.2f}")


class PerformanceMetrics:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.end_time: float = 0.0
        self.start_time: float = 0.0
        self.processed_events: int = 0
        self.total_events: int = 0
        self.row_data: List[str] = []
        self.reset()

    def reset(self) -> None:
        self.start_time = time.time()
        self.processed_events = 0

    def calculate_step(self, len_window: int, window_counter: int) -> None:
        self.end_time = time.time()

        performance: str = self._calculate_metrics()
        window_str: str = f"{window_counter},{len_window},"
        total_events_up_to_window = self.total_events + len_window
        row = window_str + performance + f",{self.total_events},{total_events_up_to_window}"
        self.row_data.append(row)
        self.total_events = total_events_up_to_window  # Update total events
        self.reset()

    def save(self, name: str = "consumer_computational_metrics") -> None:
        if len(self.row_data) == 0:
            print("No metrics to save")
            return

        filename: str = f"docs/{name}[{datetime.now()}].csv"
        with open(filename, "w") as f:
            f.write(
                "I,Window Size,Processing Time (s),Events/s,CPU (%),Memory (MB), Total Events,Cumulative Total Events\n")
            for row in self.row_data:
                f.write(row + "\n")

            print(f"Metrics saved to {filename}")

    def print(self) -> None:
        headers = ["Window Size", "Processing Time (s)", "Events/s", "CPU (%)", "Memory (MB)", "Total Events",
                   "Cumulative Total Events"]
        print("Performance Metrics")
        print(tabulate([data.split(",") for data in self.row_data], headers=headers, tablefmt="grid"))

    def _calculate_metrics(self) -> str:
        processing_time: float = self.end_time - self.start_time
        event_rate: float = self.processed_events / processing_time if processing_time > 0 else 0
        cpu_usage: float = self.process.cpu_percent(interval=None) / psutil.cpu_count()  # CPU usage in percent per core
        memory_info = self.process.memory_info()
        memory_usage: float = memory_info.rss / (1024 ** 2)  # RSS in MB
        return (
            f"{processing_time},"
            f"{event_rate},"
            f"{cpu_usage},"
            f"{memory_usage},"
        )


def calculate_conformance_metrics(window: List[Dict[str, Any]], log_eval: pd.DataFrame, n: int) -> ConformanceMetrics:
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(pd.DataFrame(window))

    fitness: dict[str, float] = pm4py.fitness_token_based_replay(log_eval, net, initial_marking, final_marking)
    precision: float = pm4py.precision_token_based_replay(log_eval, net, initial_marking, final_marking)
    gen: float = generalization_evaluator.apply(log_eval, net, initial_marking, final_marking)
    simp: float = simplicity_evaluator.apply(net)

    # if n == 7:
    #     pm4py.view_petri_net(net, initial_marking, final_marking)

    return ConformanceMetrics(fitness["log_fitness"], precision, gen, simp)


def group_traces_by_case_id(cache: List[Dict[str, Any]], key: str = "case:concept:name") -> List[List[Dict[str, Any]]]:
    grouped_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    order: List[str] = []

    for item in cache:
        group_key: str = item[key]
        if group_key not in grouped_dict:
            order.append(group_key)
        grouped_dict[group_key].append(item)

    grouped_list: List[List[Dict[str, Any]]] = [grouped_dict[group_key] for group_key in order]

    return grouped_list


def group_traces_by_case_id_dataframe(df: pd.DataFrame, key: str = "case:concept:name") -> pd.DataFrame:
    # Group the dataframe by the specified key
    grouped_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []

    for _, row in df.iterrows():
        group_key: str = row[key]
        if group_key not in grouped_dict:
            order.append(group_key)
        grouped_dict[group_key].append(row.to_dict())

    # Creating a new DataFrame with the grouped results
    grouped_data: List[Dict[str, Any]] = [{"group": group_key, "traces": grouped_dict[group_key]} for group_key in
                                          order]
    grouped_df: pd.DataFrame = pd.DataFrame(grouped_data)

    return grouped_df


def get_tenth_percentile_index(df: pd.DataFrame) -> List[int]:
    num_rows: int = len(df)
    percentiles: List[int] = [int(num_rows * (p / 100)) for p in range(10, 100, 10)]
    percentile_indices: List[int] = [df.index[pos] for pos in percentiles]
    return percentile_indices


def combine_figures(figure_list: list[plt.Figure], max_title_length: int = 20) -> plt.Figure:
    """
    Combine multiple Matplotlib figures into a single figure with subplots arranged in two columns.
    Titles will wrap if they exceed the specified maximum length, and a single legend will be created for all subplots,
    ensuring that duplicate labels appear only once.

    :param figure_list: List of Matplotlib figure objects.
    :param max_title_length: Maximum number of characters in titles before they wrap to a new line.
    :return: A new Matplotlib figure containing subplots for each input figure, arranged in two columns.
    """
    # Determine the number of rows needed for two figures per row
    n = len(figure_list)
    rows = (n + 1) // 2  # Add one if odd number of figures

    # Create a new figure with subplots arranged in two columns
    combined_fig: plt.Figure
    combined_fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 4))

    # Flatten axes for easy indexing if needed (in case it's multi-dimensional)
    axes = axes.flatten()

    # A list to gather all handles and labels for the combined legend
    handles = []
    labels = set()  # Using a set to track unique labels

    # Copy each figure's axes and content to the corresponding subplot
    for i, fig in enumerate(figure_list):
        for ax in fig.get_axes():  # Get all axes from the original figure
            # Plot the lines on the combined figure
            axes[i].set_ylim(ax.get_ylim())

            for line in ax.get_lines():
                # axes[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
                axes[i].axvline(5000, color="Green", linewidth=4, ymax=5)

            # Set the title and wrap if needed
            title = ax.get_title()
            if len(title) > max_title_length:
                wrapped_title = '\n'.join(
                    [title[i:i + max_title_length] for i in range(0, len(title), max_title_length)])
                axes[i].set_title(wrapped_title)
            else:
                axes[i].set_title(title)

            axes[i].set_xlabel(ax.get_xlabel())
            axes[i].set_ylabel(ax.get_ylabel())

            # Collect unique handles and labels for the legend
            for handle, label in zip(*axes[i].get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.add(label)

    # Hide any unused axes (if there are odd number of figures)
    for j in range(i + 1, len(axes)):
        combined_fig.delaxes(axes[j])

    # Add a single legend for all subplots at the bottom, with unique labels only
    combined_fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to fit the legend
    combined_fig.savefig("species_evaluation1.png", bbox_inches='tight')
    return combined_fig
