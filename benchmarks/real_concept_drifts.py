import pandas as pd
import pm4py

from src.estimator_accumulative import WindowEstimator, SpeciesRetrievalRepository
from src.utils.methods import ConformanceMetrics
from typing import List, Dict, Any

# LOG: pd.DataFrame = pm4py.read_xes("/Users/christianimenkamp/Documents/Data-Repository/Concept Drifts/bose_log.xes")

PATH = "/Users/christianimenkamp/Documents/Data-Repository/Concept Drifts/recurring[3Changes_].xes"
LOG: pd.DataFrame = pm4py.read_xes(PATH)

conformance_metrics: List[tuple[int, None, tuple[str, str]]] = []
window_counter: int = 0
#
#
def on_window(log: List[Dict[str, Any]], index: int) -> None:
    global window_counter
    global log_counter
    global LOGS
    window_counter += 1
    e_1 = log[0]["concept:name"]
    e_2 = log[-1]["concept:name"]

    conformance_metrics.append((len(log), None, (e_1, e_2)))


window: WindowEstimator = WindowEstimator(
    on_full_completeness=on_window,
    species=SpeciesRetrievalRepository.retrieve_species_1_gram,
)

for i, row in LOG.iterrows():
    window.add_event(row.to_dict())
    window.get_metrics()

results = pd.DataFrame(conformance_metrics)
print(results)
results[0].plot()

results.to_csv("concept_drifts_real/recurring[3Changes_].csv")





