from functools import partial

import pm4py
import time

from src.completeness.species_estimator import SpeciesEstimator, METRICS
from typing import Optional, Union, List, Callable, TypeVar, Dict, Any

from src.completeness.species_retrieval import retrieve_species_n_gram
from src.utils.methods import ConformanceMetrics, calculate_conformance_metrics
import warnings

warnings.filterwarnings('ignore')


class WindowEstimator:
    def __init__(self,
                 completeness_threshold: float = 0.90,
                 initial_data: List[Dict[str, Any]] = None,
                 on_full_completeness: Callable[[List[Dict[str, Any]], int], None] = None) -> None:

        if initial_data:
            self._window: List[Dict[str, Any]] = initial_data
        else:
            self._window: List[Dict[str, Any]] = []

        self.completeness_threshold = completeness_threshold
        self.completeness_cache: List[List[int]] = []
        self.time_cache: List[tuple[int, float]] = []
        self.conformance_cache: List[tuple[int, ConformanceMetrics]] = []
        self.on_full_completeness = on_full_completeness
        self._window_counter: int = 0

    def add_event(self, row_data: Dict[str, Any]) -> None:

        self._window.append(row_data)

    def get_metrics(self, DEBUG: bool = False) -> float:
        if DEBUG:
            start_time: float = time.time()

        estimator = SpeciesEstimator(partial(retrieve_species_n_gram, n=1), quantify_all=True)
        estimator.profile_log(self._window)

        estimated: float = getattr(estimator, "coverage_incidence")[0]
        #estimated: float = getattr(estimator, "completeness_incidence")[0]

        if len(self._window) <= 5:
            estimated = 0.0

        if DEBUG:
            self.time_cache.append((len(self._window), float(time.time() - start_time) * 1000))

        self.completeness_cache.append([len(self._window), estimated])

        if (self.on_full_completeness is not None
                and estimated >= self.completeness_threshold):
            self._window_counter += 1
            self.on_full_completeness(self._window, self._window_counter)
            self._window.clear()

        return estimated

    def get_window(self) -> List[Dict[str, Any]]:
        return self._window
