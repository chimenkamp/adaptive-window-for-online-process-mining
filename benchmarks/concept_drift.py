import os

import pm4py

from src.estimator_accumulative import WindowEstimator
from src.utils.plotter import Plotter
import pandas as pd
from typing import List
from typing import List, Dict, Any
from src.utils.methods import ConformanceMetrics


def get_files_ending_with_1000_xes(folder_path: str) -> List[str]:
    """
    Returns the paths of all files in the folder that end with '-1000.xes'.

    :param folder_path: The path to the folder containing the files.
    :return: A list of file paths that match the pattern '*-1000.xes'.
    """
    matching_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith("-10000.xes"):
            full_path = os.path.join(folder_path, file_name)
            matching_files.append(full_path)

    return matching_files


conformance_metrics: List[tuple[int, ConformanceMetrics | None, tuple[str, str]]] = []

CONCEPT_DRIFT_PATH = '/Users/christianimenkamp/Documents/Data-Repository/Concept Drifts/'


def on_full_completeness(window: List[Dict[str, Any]], window_counter: int) -> None:
    e_1 = window[0]["concept:name"]
    e_2 = window[-1]["concept:name"]
    conformance_metrics.append((len(window), None, (e_1, e_2)))


filename_mapping = {}

for file_path in get_files_ending_with_1000_xes(CONCEPT_DRIFT_PATH):
    df = pm4py.read_xes(file_path)
    filename_mapping[f"{file_path.split('/')[-1].replace('.xes', '.csv')}"] = len(df)
    # print(file_path)
    # conformance_metrics.clear()
    # LOG: pd.DataFrame = pm4py.read_xes(file_path)
    # START_AT, STOP_AT = (0, len(LOG) - 1)
    #
    # window = WindowEstimator(
    #     on_full_completeness=on_full_completeness
    # )
    #
    # for i, row in LOG.iterrows():
    #     window.add_event(row.to_dict())
    #
    #     if START_AT < i <= STOP_AT:
    #         window.get_metrics(DEBUG=False)
    #
    #     if i == STOP_AT + 1:
    #         break
    #
    # RESULT_DF: pd.DataFrame = pd.DataFrame(conformance_metrics,
    #                                        columns=["No. of events", "Conformance Metrics", "Event Pair"])
    #
    # RESULT_DF.to_csv(f"concept_drift_synthetic/{file_path.split('/')[-1].replace('.xes', '.csv')}")

print(filename_mapping)
