import time
from functools import partial
from typing import List, Dict, Any

import pandas as pd
import pm4py

from src.completeness.species_estimator import SpeciesEstimator
from src.completeness.species_retrieval import retrieve_species_n_gram
from src.estimator_accumulative import WindowEstimator
from src.utils.plotter import Plotter
from src.utils.methods import (
    print_progress_bar,
    find_non_zero_ranges,
    calculate_conformance_metrics,
    ConformanceMetrics)
from src.utils.windowing_baseline import CountBasedWindow

COMPLETENESS_THRESHOLD: float = 0.65
DEBUG: bool = False

conformance_metrics: List[tuple[int, ConformanceMetrics | None, tuple[str, str]]] = []

DATA_PATH: str = "data/"
# LOG_NAME: str = "Community/sepsis/Sepsis Cases - Event Log.feather"
# LOG_NAME: str = "Community/daily_living/activity_log.xes"
# LOG_NAME: str = "Synthetic/synthetic, online order/online_order.xes"

# LOG_NAME: str = "Community/bpi-c-2012/BPI_Challenge_2012.feather"
# LOG_NAME: str = "Concept_Drift/Synthetic Event Logs for Concept Drift Detection/cb-2500.feather"
LOG_NAME: str = ("/Merged/sudden_drift[bpi-c-2012,bpi-c-2013,bpi-c-2015,bpi-c-2018,bpi-c-2017,"
                 "bpi-c-2019].feather")

LOG_EVAL_FULL: pd.DataFrame = pd.read_feather(DATA_PATH + LOG_NAME)

# Select 25% of the data
# LOG_EVAL = LOG_EVAL_FULL[:int(len(LOG_EVAL_FULL) * 0.125)]
LOG_EVAL = LOG_EVAL_FULL
print(len(LOG_EVAL))


def on_full_completeness(window: List[Dict[str, Any]], window_counter: int) -> None:
    e_1 = window[0]["concept:name"]
    e_2 = window[-1]["concept:name"]
    if DEBUG:
        metrics: ConformanceMetrics = calculate_conformance_metrics(window, LOG_EVAL, window_counter)
        conformance_metrics.append((len(window), metrics, (e_1, e_2)))
    else:
        conformance_metrics.append((len(window), None, (e_1, e_2)))


window = WindowEstimator(
    completeness_threshold=COMPLETENESS_THRESHOLD,
    on_full_completeness=on_full_completeness)


START_AT, STOP_AT = (0, len(LOG_EVAL)-1)

for i, row in LOG_EVAL.iterrows():
    window.add_event(row.to_dict())

    if START_AT < i <= STOP_AT:
        window.get_metrics(DEBUG=DEBUG)

    if i == STOP_AT + 1:
        break

    print_progress_bar(i, STOP_AT, prefix='Progress:', suffix='Complete', length=50)

x_axis_range: List[int] = list(range(START_AT, STOP_AT))

plotter = Plotter(x_axis_range)

full_completeness: list[int] = [y[1] for y in window.completeness_cache]
full_time: list[float] = [y[1] for y in window.time_cache]
full_conformance: list[ConformanceMetrics] = [y[1] for y in conformance_metrics]

plotter.add_subplot(
    [
        (f"Completeness (threshold={COMPLETENESS_THRESHOLD})", full_completeness),
    ]
)

if DEBUG:
    plotter.add_subplot([(f"Time", full_time)])

for i, block in enumerate(find_non_zero_ranges(full_completeness)):
    try:
        block_fit: tuple[int, ConformanceMetrics, tuple[str, str]] = conformance_metrics[i]
        plotter.shade_regions(
            block[0],
            block[1],
            alpha=0.3,
            text=f"No. of events: {block_fit[0]}, {str(block_fit[1]) if DEBUG else ''}",
            position="middle",
            orientation="vertical",
        )
    except IndexError:
        pass

if DEBUG:
    R: int = 5
    display: List[List[float]] = [
        [
            float(i + 1),
            x[0],
            round(x[1].f1_score, R),
            round(x[1].fitness, R),
            round(x[1].generalization, R),
            round(x[1].precision, R),
            round(x[1].simplicity, R),
            x[2]
        ]
        for i, x in enumerate(conformance_metrics)
    ]

    plotter.add_table(
        display,
        headers=["I", "N", "F1 Score", "Fitness", "Generalization", "Precision", "Simplicity", "Events"],
        condition=lambda x: 'lightgreen' if isinstance(x, float)
                                            and (0.9 <= x < 1.0) else '#FFCCCB' if isinstance(x, float)
                                                                                   and (x < 0.7) else None,
    )


# for idx in get_tenth_percentile_index(LOG_EVAL_FULL)[:1]:
#     print(idx)
#     plotter.draw_vertical_line(idx, "#8B0000", 5)

plotter.draw_vertical_line(644, "#8B0000", 5, label="<- bpi-c-2012", label_position="top")
plotter.draw_vertical_line(769, "#8B0000", 5, label="<- bpi-c-2013", label_position="top")
plotter.draw_vertical_line(1881, "#8B0000", 5, label="<- bpi-c-2015", label_position="top")
plotter.draw_vertical_line(3098, "#8B0000", 5, label="<- bpi-c-2018", label_position="top")
plotter.draw_vertical_line(3898, "#8B0000", 5, label="<- bpi-c-2017", label_position="top")
# plotter.draw_vertical_line(2825, "#8B0000", 5, label="Sudden Concept Drift", label_position="middle")

plotter.plot(50,20)
