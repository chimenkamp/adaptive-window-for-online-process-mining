from datetime import timedelta
from typing import List, Dict, Any, Tuple

import pandas as pd

from src.estimator_accumulative import WindowEstimator, SpeciesRetrievalRepository
from src.utils.methods import ConformanceMetrics, calculate_conformance_metrics, group_traces_by_case_id, \
    group_traces_by_case_id_dataframe
from src.utils.plotter import Plotter
from src.utils.windowing_baseline import (TimeBasedTumblingWindow, AbstractWindow, LandmarkWindow, LossyCounting,
                                          TimeBasedSlidingWindow, LossyCountingWithBudget)

conformance_metrics: List[tuple[int, ConformanceMetrics, tuple[str, str]]] = []

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
window_counter: int = 0
log_counter: int = 0


def conformance_metrics_to_dataframe(
        metrics_list: List[Tuple[int, ConformanceMetrics, Tuple[str, str]]], log_name: str, species_name: str, window_counter: int
) -> pd.DataFrame:
    """
    Transforms a list of tuples containing conformance metrics and related information into a pandas DataFrame.

    :param metrics_list: List of tuples with the format:
                         (log_length, ConformanceMetrics object, (start_event, end_event)).
    :return: DataFrame with columns ['Log Length', 'Precision', 'Fitness', 'F1-Score', 'Start Event', 'End Event'].
    """
    data = []

    for log_length, metrics, (start_event, end_event) in metrics_list:
        data.append({
            "Log Name": log_name,
            "Species": species_name,
            "Window Counter": window_counter,
            "Window Size": log_length,
            "Precision": metrics.precision,
            "Fitness": metrics.fitness,
            "F1-Score": metrics.f1_score,
            "Start Event": start_event,
            "End Event": end_event,
        })

    return pd.DataFrame(data)

def on_window(log: List[Dict[str, Any]], index: int) -> None:
    global window_counter
    global log_counter
    global LOGS
    window_counter += 1
    e_1 = log[0]["concept:name"]
    e_2 = log[-1]["concept:name"]
    metrics: ConformanceMetrics = calculate_conformance_metrics(log, LOGS[log_counter][1], window_counter)
    conformance_metrics.append((len(log), metrics, (e_1, e_2)))


def print_min_max_avg(lis: List[float]):
    print("MIN: ", min(lis), "MAX: ", max(lis), "AVG: ", sum(lis) / len(lis))


def get_min_max_avg(lis: List[float]):
    if lis == [] or lis is None or len(lis) == 0:
        return 0, 0, 0
    return min(lis), max(lis), sum(lis) / len(lis)


# For LossyCounting with epsilon = 0.01
# window: AbstractWindow = LossyCounting(0.05, on_window=on_window)

# For LossyCountingWithBudget with epsilon = 0.01 and budget = 500
# window: AbstractWindow = LossyCountingWithBudget(0.05, 10, on_window=on_window)

# For LandmarkWindow with landmark = "landmark_event"
# window: AbstractWindow = LandmarkWindow("ER Registration", on_window=on_window)

# For TimeBasedSlidingWindow with window_size = timedelta(conds=5)
# window: AbstractWindow = TimeBasedSlidingWindow(timedelta(seconds=5), on_window=on_window)

# For TimeBasedTumblingWindow with window_size = timedelta(seconds=5)
# window: AbstractWindow = TimeBasedTumblingWindow(timedelta(hours=12), on_window=on_window)

# For EstimatorWindow with completeness_threshold = 0.90

all_metrics: pd.DataFrame = pd.DataFrame()
repository = SpeciesRetrievalRepository()

for LOG in LOGS:

    for species in repository.get_all():
        print("Species: ", species)
        conformance_metrics.clear()
        window: WindowEstimator = WindowEstimator(
            on_full_completeness=on_window,
            species=species
        )

        for i, row in LOG[1].iterrows():
            window.add_event(row.to_dict())
            window.get_metrics()
        window_counter = 0
        conformance_metrics_to_dataframe(conformance_metrics, LOG[0], species[0], window_counter).to_csv(
            f"species_windows/{LOG[0]}_{species[0]}_window_metrics.csv")

        table_data = [
            list(
                get_min_max_avg([conformance_metrics[i][1].precision for i in range(len(conformance_metrics))])),
            list(
                get_min_max_avg([conformance_metrics[i][1].fitness for i in range(len(conformance_metrics))])),
            list(
                get_min_max_avg([conformance_metrics[i][1].f1_score for i in range(len(conformance_metrics))]))
        ]

        metrics_combined = pd.DataFrame(
            [[LOG[0], species[0]] + list(table_data[0]) + list(table_data[1]) + list(table_data[2])],
            columns=[
                "Log Name", "Species",
                "Precision Min", "Precision Max", "Precision Avg",
                "Fitness Min", "Fitness Max", "Fitness Avg",
                "F1-Score Min", "F1-Score Max", "F1-Score Avg"
            ]
        )

        all_metrics = pd.concat([all_metrics, metrics_combined], ignore_index=True)

    log_counter += 1

all_metrics.to_csv("landmark_sepsis.csv")
