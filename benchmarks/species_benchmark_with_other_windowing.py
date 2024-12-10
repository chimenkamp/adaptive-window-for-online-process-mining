from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Tuple
from datetime import datetime, timedelta
import pandas as pd
import os

from src.utils.methods import ConformanceMetrics, calculate_conformance_metrics

DATA_PATH: str = "/Users/christianimenkamp/Documents/Data-Repository/Community/"
SYNTHETIC_PATH: str = "/Users/christianimenkamp/Documents/Data-Repository/Synthetic/"

num_events: int = 2000

A: tuple[str, pd.DataFrame] = ("bpi-c-2012", pd.read_feather(DATA_PATH + "/bpi-c-2012/data.feather")[0:num_events])
B: tuple[str, pd.DataFrame] = ("bpi-c-2013", pd.read_feather(DATA_PATH + "/bpi-c-2013/data.feather")[0:num_events])
C: tuple[str, pd.DataFrame] = ("bpi-c-2015", pd.read_feather(DATA_PATH + "/bpi-c-2015/data.feather")[0:num_events])
D: tuple[str, pd.DataFrame] = (
    "Sepsis Cases", pd.read_feather(DATA_PATH + "sepsis/Sepsis Cases - Event Log.feather")[0:num_events])
E: tuple[str, pd.DataFrame] = (
    "hospital-billing", pd.read_feather(DATA_PATH + "hospital-billing/hospital-billing.feather")[0:num_events])
F: tuple[str, pd.DataFrame] = (
    "synthetic, online order", pd.read_feather(SYNTHETIC_PATH + "synthetic, online order/online_order.feather")[0:num_events])
G: tuple[str, pd.DataFrame] = ("bpi-c-2015", pd.read_feather(DATA_PATH + "/bpi-c-2015/data.feather")[0:num_events])
H: tuple[str, pd.DataFrame] = ("bpi-c-2017", pd.read_feather(DATA_PATH + "/bpi-c-2017/data.feather")[0:num_events])
I: tuple[str, pd.DataFrame] = ("Road-Traffic-Fine-Management-Process", pd.read_feather(
    DATA_PATH + "Road-Traffic-Fine-Management-Process/Road_Traffic_Fine_Management_Process.feather")[0:num_events])

LOGS = [A, B, C, D, E, F, G, H, I]

class AbstractWindow(ABC):
    """Abstract base class for all windowing strategies."""

    def __init__(self, on_window_callback: Callable[[List[Dict[str, Any]]], None]):
        self.window: List[Dict[str, Any]] = []
        self.on_window_callback = on_window_callback

    @abstractmethod
    def observe_event(self, event: Dict[str, Any]) -> None:
        """Process a new event according to the windowing strategy."""
        pass

    def _emit_window(self) -> None:
        """Emit the current window if not empty and call the callback."""
        if self.window:
            self.on_window_callback(self.window.copy())


class CountBasedTumblingWindow(AbstractWindow):
    """Implementation of a count-based tumbling window."""

    def __init__(self, window_size: int, on_window_callback: Callable[[List[Dict[str, Any]]], None]):
        super().__init__(on_window_callback)
        self.window_size = window_size

    def observe_event(self, event: Dict[str, Any]) -> None:
        self.window.append(event)

        if len(self.window) >= self.window_size:
            self._emit_window()
            self.window.clear()


class LandmarkWindow(AbstractWindow):
    """Implementation of a landmark window that accumulates all events."""

    def __init__(self, on_window_callback: Callable[[List[Dict[str, Any]]], None]):
        super().__init__(on_window_callback)

    def observe_event(self, event: Dict[str, Any]) -> None:
        self.window.append(event)
        self._emit_window()


class TimeBasedTumblingWindow(AbstractWindow):
    """Implementation of a time-based tumbling window."""

    def __init__(self, window_size: timedelta, on_window_callback: Callable[[List[Dict[str, Any]]], None]):
        super().__init__(on_window_callback)
        self.window_size = window_size
        self.window_start: datetime | None = None

    def observe_event(self, event: Dict[str, Any]) -> None:
        timestamp = event.get('time:timestamp')
        if not isinstance(timestamp, datetime):
            raise ValueError("Event must have a valid timestamp")

        if self.window_start is None:
            self.window_start = timestamp

        if timestamp - self.window_start >= self.window_size:
            self._emit_window()
            self.window.clear()
            self.window_start = timestamp

        self.window.append(event)


class MetricsManager:
    """Manages the collection and storage of metrics."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.all_metrics = pd.DataFrame()
        self.window_metrics: List[Tuple[int, ConformanceMetrics, Tuple[str, str]]] = []
        os.makedirs(output_dir, exist_ok=True)

    def clear_window_metrics(self):
        """Clear the current window metrics."""
        self.window_metrics.clear()

    def add_window_metric(self, log_length: int, metrics: ConformanceMetrics, events: Tuple[str, str]):
        """Add a new window metric."""
        self.window_metrics.append((log_length, metrics, events))

    def save_window_metrics(self, log_name: str, species: str, window_counter: int):
        """Save the current window metrics to a CSV file."""
        df = self._window_metrics_to_dataframe(log_name, species, window_counter)
        filename = f"{self.output_dir}/{species.lower()}_{log_name.lower()}_windows.csv"
        df.to_csv(filename, index=False)
        return df

    def save_summary_metrics(self, filename: str = "all_metrics_summary.csv"):
        """Save the summary metrics to a CSV file."""
        self.all_metrics.to_csv(os.path.join(self.output_dir, filename), index=False)

    def add_summary_metrics(self, log_name: str, species: str):
        """Add summary metrics for the current window set."""
        precision_stats = self._get_metric_stats('precision')
        fitness_stats = self._get_metric_stats('fitness')
        f1_stats = self._get_metric_stats('f1_score')

        metrics_combined = pd.DataFrame(
            [[log_name, species] + precision_stats + fitness_stats + f1_stats],
            columns=[
                "Log Name", "Species",
                "Precision Min", "Precision Max", "Precision Avg",
                "Fitness Min", "Fitness Max", "Fitness Avg",
                "F1-Score Min", "F1-Score Max", "F1-Score Avg"
            ]
        )
        self.all_metrics = pd.concat([self.all_metrics, metrics_combined], ignore_index=True)

    def _window_metrics_to_dataframe(self, log_name: str, species: str, window_counter: int) -> pd.DataFrame:
        """Convert window metrics to DataFrame."""
        data = []
        for log_length, metrics, (start_event, end_event) in self.window_metrics:
            data.append({
                "Log Name": log_name,
                "Species": species,
                "Window Counter": window_counter,
                "Window Size": log_length,
                "Precision": metrics.precision,
                "Fitness": metrics.fitness,
                "F1-Score": metrics.f1_score,
                "Start Event": start_event,
                "End Event": end_event,
            })
        return pd.DataFrame(data)

    def _get_metric_stats(self, metric_name: str) -> List[float]:
        """Calculate min, max, avg statistics for a given metric."""
        values = [getattr(m[1], metric_name) for m in self.window_metrics]
        if not values:
            return [0.0, 0.0, 0.0]
        return [min(values), max(values), sum(values) / len(values)]


def run_benchmark(log_data: pd.DataFrame, log_name: str, window_type: str,
                  metrics_manager: MetricsManager, **window_params) -> None:
    """
    Run benchmarking with the specified window type.

    Args:
        log_data: DataFrame containing the log data
        log_name: Name of the log being processed
        window_type: Type of window to use ('count', 'landmark', or 'time')
        metrics_manager: MetricsManager instance for handling metrics
        **window_params: Parameters for window initialization
    """
    window_counter = 0
    metrics_manager.clear_window_metrics()

    def on_window(window_data: List[Dict[str, Any]]) -> None:
        nonlocal window_counter
        window_counter += 1
        e_1 = window_data[0]["concept:name"]
        e_2 = window_data[-1]["concept:name"]
        metrics = calculate_conformance_metrics(window_data, log_data, window_counter)
        metrics_manager.add_window_metric(len(window_data), metrics, (e_1, e_2))

    # Initialize the appropriate window type
    if window_type == "count":
        window = CountBasedTumblingWindow(
            window_size=window_params.get('window_size', 50),
            on_window_callback=on_window
        )
        species = "Count-Based"
    elif window_type == "landmark":
        window = LandmarkWindow(on_window_callback=on_window)
        species = "Landmark"
    elif window_type == "time":
        window = TimeBasedTumblingWindow(
            window_size=window_params.get('window_size', timedelta(hours=1)),
            on_window_callback=on_window
        )
        species = "Time-Based"
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    # Process events through the window
    for _, row in log_data.iterrows():
        event = row.to_dict()
        window.observe_event(event)

    # Save metrics
    metrics_manager.save_window_metrics(log_name, species, window_counter)
    metrics_manager.add_summary_metrics(log_name, species)


def main():
    # Initialize metrics manager
    metrics_manager = MetricsManager(output_dir="benchmark_results_windowing")

    # Define window configurations
    window_configs = [
        ("count", {"window_size": 20}),
        ("landmark", {"concept:name": "ER Registration"}),
        ("time", {"window_size": timedelta(hours=1)})
    ]

    # Process each log with each window type
    for LOG in [D]:
        print(f"Processing log: {LOG[0]}")
        for window_type, params in window_configs:
            print(f"Running {window_type} window...")
            run_benchmark(
                log_data=LOG[1],
                log_name=LOG[0],
                window_type=window_type,
                metrics_manager=metrics_manager,
                **params
            )

    # Save final summary metrics
    metrics_manager.save_summary_metrics()


if __name__ == "__main__":
    main()