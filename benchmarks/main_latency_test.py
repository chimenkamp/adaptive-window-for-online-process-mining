import cProfile
import pstats
from functools import partial
from typing import Any, Dict, List
import pandas as pd
import pm4py
import perfplot
from matplotlib import pyplot as plt
from pm4py.objects.log.obj import EventLog

from src.clean_window_estimator import WindowEstimatorClean
from src.utils.plotter import Plotter
from src.window_estimator import WindowEstimatorDebug
import pstats

plt.rcParams["figure.autolayout"] = True
plt.style.use('seaborn-v0_8-darkgrid')

window = WindowEstimatorClean[Dict[str, Any]](
    initial_data=[],
    on_full_completeness=lambda x, y: None,
)

# Function to calculate completeness
def get_completeness(events: List[Dict[str, Any]]) -> float:
    for event in events:
        window.add_event(event)
    return window.get_metrics()


# Constants
DATA_PATH: str = "data/"
LOG_NAME: str = "/Users/christianimenkamp/Documents/Data-Repository/Community/sepsis/Sepsis Cases - Event Log.feather"

if __name__ == "__main__":
    p = pstats.Stats('../profile_stats.prof')

    # Make the output more readable
    p.strip_dirs()

    # Sort the statistics by cumulative time spent in a function
    p.sort_stats(pstats.SortKey.CUMULATIVE)

    # Print the statistics (top 10 functions)
    p.print_stats(10)
    p.
