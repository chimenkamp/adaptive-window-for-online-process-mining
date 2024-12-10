import cProfile
import pstats
from functools import partial
from typing import Any, Dict, List
import pandas as pd
import pm4py
import perfplot
from matplotlib import pyplot as plt
from pm4py.objects.log.obj import EventLog

from src.completeness.species_estimator import SpeciesEstimator
from src.completeness.species_retrieval import retrieve_species_n_gram
from src.utils.plotter import Plotter

plt.rcParams["figure.autolayout"] = True
plt.style.use('seaborn-v0_8-darkgrid')


# Function to calculate completeness
def get_completeness(events: List[Dict[str, Any]]) -> float:
    estimator = SpeciesEstimator(partial(retrieve_species_n_gram, n=1), quantify_all=True)
    estimator.profile_log(events)
    estimated: float = getattr(estimator, "completeness_incidence")[0]
    return estimated


# Constants
DATA_PATH: str = "data/"
# LOG_NAME: str = "Community/sepsis/Sepsis Cases - Event Log.feather"
# LOG_NAME: str = "Community/daily_living/activity_log.feather"
# LOG_NAME: str = "Synthetic/synthetic, online order/online_order.feather"
# LOG_NAME: str = "Community/bpi-c-2012/BPI_Challenge_2012.feather"
LOG_NAME: str = "sepsis/Sepsis Cases - Event Log.feather"

if __name__ == "__main__":
    df_log: pd.DataFrame = pd.read_feather(DATA_PATH + LOG_NAME)
    event_log: List[Dict[str, Any]] = df_log.to_dict(orient='records')

    # perfplot.show(
    #     setup=lambda n: event_log[:n],
    #     kernels=[get_completeness],
    #     n_range=range(1, min(100, len(event_log))),
    #     xlabel='Number of Events',
    #     equality_check=None,
    #     title='Completeness Estimation Performance',
    #     time_unit='ms',
    #     logx=True,
    #     logy=True,
    # )

    g = perfplot.bench(
        setup=lambda n: event_log[:n],
        kernels=[get_completeness],
        n_range=list(range(1, min(550, len(event_log)))),
        equality_check=None,
    )

    n = list(g.n_range)
    t = list(g.timings_s)[0]

    plotter = Plotter(n, "")
    plotter.add_subplot([("Runtime [s]", t)])
    plotter.plot(x_label="Number of Events", y_label="Runtime [S]", font_size=28)
