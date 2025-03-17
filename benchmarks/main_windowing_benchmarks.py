from datetime import timedelta
from typing import List, Dict, Any

import pandas as pd

from src.utils.methods import ConformanceMetrics, calculate_conformance_metrics, group_traces_by_case_id, \
    group_traces_by_case_id_dataframe
from src.utils.plotter import Plotter
from src.utils.windowing_baseline import (TimeBasedTumblingWindow, AbstractWindow, LandmarkWindow, LossyCounting,
                                          TimeBasedSlidingWindow, LossyCountingWithBudget)
from src.window_estimator import SpeciesRetrievalRepository, WindowEstimatorDebug

conformance_metrics: List[tuple[int, ConformanceMetrics, tuple[str, str]]] = []

DATA_PATH: str = "/Users/christianimenkamp/Documents/Data-Repository/Community/"
# LOG_EVAL_STR: str = "Concept_Drift/Merged/sudden_drift_short[bpi-c-2012,bpi-c-2013,bpi-c-2015].feather"

# LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "/Community/bpi-c-2012/data.feather")[0:1000]
# LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "/Community/bpi-c-2013/data.feather")[0:1000]
# LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "/Community/bpi-c-2015/data.feather")[0:1000]
# LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "hospital-billing/hospital-billing.feather")[0:1000]
LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "hospital-billing/hospital-billing.feather")[0:1000]
# LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "Synthetic/synthetic, online order/online_order.xes")[0:1000]
# LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "/Community/bpi-c-2015/data.feather")[0:1000]
# LOG_EVAL: pd.DataFrame = pd.read_feather(DATA_PATH + "/Community/bpi-c-2015/data.feather")[0:1000]


window_counter: int = 0


def on_window(log: List[Dict[str, Any]], index: int) -> None:
    global window_counter
    window_counter += 1
    e_1 = log[0]["concept:name"]
    e_2 = log[-1]["concept:name"]
    metrics: ConformanceMetrics = calculate_conformance_metrics(log, LOG_EVAL, window_counter)
    conformance_metrics.append((len(log), metrics, (e_1, e_2)))


def print_min_max_avg(lis: List[float]):
    print("MIN: ", min(lis), "MAX: ", max(lis), "AVG: ", sum(lis) / len(lis))


def get_min_max_avg(lis: List[float]):
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


repository = SpeciesRetrievalRepository()

for species in repository.get_all():
    print("Species: ", species)
    conformance_metrics.clear()
    window: WindowEstimatorDebug = WindowEstimatorDebug[Dict](
        on_full_completeness=on_window,
        species=species
    )

    for i, row in LOG_EVAL.iterrows():
        window.add_event(row.to_dict())
        window.get_metrics()

    # title: str = "LossyCounting (epsilon=0.05) on Sepsis Cases - Event Log"
    # title: str = "LossyCountingWithBudget (epsilon=0.05, budget=10) on Sepsis Cases - Event Log"
    # title: str = "LandmarkWindow (landmark=ER Registration) on Sepsis Cases - Event Log"
    # title: str = "TimeBasedTumblingWindow (window_size=12 hours) on Sepsis Cases - Event Log"
    name: str = "hospital - billing_" + species[0]
    title: str = f"Species Estimator with {name} - Event Log"

    print("precision")
    print_min_max_avg([conformance_metrics[i][1].precision for i in range(len(conformance_metrics))])
    print("fitness")
    print_min_max_avg([conformance_metrics[i][1].fitness for i in range(len(conformance_metrics))])
    print("f1_score")
    print_min_max_avg([conformance_metrics[i][1].f1_score for i in range(len(conformance_metrics))])

    plotter = Plotter(list(range(0, len(conformance_metrics))), title)
    plotter.add_subplot(
        [
            ("F1 Score", [conformance_metrics[i][1].f1_score for i in range(len(conformance_metrics))]),
            ("Fitness", [conformance_metrics[i][1].fitness for i in range(len(conformance_metrics))]),
            ("Precision", [conformance_metrics[i][1].precision for i in range(len(conformance_metrics))]),
            # ("Simplicity", [conformance_metrics[i][1].simplicity for i in range(len(conformance_metrics))]),
            # ("Generalization", [conformance_metrics[i][1].generalization for i in range(len(conformance_metrics))])
        ]
    )
    plotter.add_table(
        table_data=[
            ["Precision"] + list(get_min_max_avg([conformance_metrics[i][1].precision for i in range(len(conformance_metrics))])),
            ["Fitness"] + list(get_min_max_avg([conformance_metrics[i][1].fitness for i in range(len(conformance_metrics))])),
            ["F1-Score"] + list(get_min_max_avg([conformance_metrics[i][1].f1_score for i in range(len(conformance_metrics))]))
        ],
        headers=["Metric", "Min ", "Max", "Average"]
    )
    plotter.plot(x_label="Window", save_path="/Users/christianimenkamp/Documents/Git-Repositorys/Event Stream Window - Completeness/figures/species/" + name + ".png")
