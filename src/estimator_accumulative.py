import dataclasses
from functools import partial

import pandas as pd
import pm4py
import time

from typing import Optional, Union, List, Callable, TypeVar, Dict, Any

from special4pm.estimation import SpeciesEstimator

from special4pm.estimation.metrics import coverage

from special4pm.species import (
    retrieve_species_n_gram,
    retrieve_species_trace_variant,
    retrieve_timed_activity,
    retrieve_timed_activity_exponential
)

from src.utils.dynamic_threshold_estimator import DynamicThresholdEstimator
from src.utils.methods import ConformanceMetrics, calculate_conformance_metrics
import warnings

warnings.filterwarnings('ignore')
SpeciesDef = Callable[[pd.DataFrame], pd.DataFrame]
SpeciesRetrivalDef = tuple[str, SpeciesDef]


@dataclasses.dataclass
class SpeciesRetrievalRepository:
    retrieve_species_1_gram: SpeciesRetrivalDef = ("1-gram", partial(retrieve_species_n_gram, n=1))
    retrieve_species_2_gram: SpeciesRetrivalDef = ("2-gram", partial(retrieve_species_n_gram, n=2))
    retrieve_species_3_gram: SpeciesRetrivalDef = ("3-gram", partial(retrieve_species_n_gram, n=3))
    retrieve_species_4_gram: SpeciesRetrivalDef = ("4-gram", partial(retrieve_species_n_gram, n=4))
    retrieve_species_5_gram: SpeciesRetrivalDef = ("5-gram", partial(retrieve_species_n_gram, n=5))

    trace_varints: SpeciesRetrivalDef = ("trace_variant", retrieve_species_trace_variant)

    retrieve_timed_activity: SpeciesRetrivalDef = ("timed_activity", partial(retrieve_timed_activity, interval_size=2))
    retrieve_timed_activity_exponential: SpeciesRetrivalDef \
        = ("timed_activity_exponential", retrieve_timed_activity_exponential)

    def get_all(self) -> List[SpeciesRetrivalDef]:
        """
        Retrieve all species retrieval definitions.

        :return: List of all SpeciesRetrivalDef objects from the class instance.
        """
        return [getattr(self, field.name) for field in dataclasses.fields(self)]


class WindowEstimator:
    def __init__(self,
                 initial_data: List[Dict[str, Any]] = None,
                 on_full_completeness: Callable[[List[Dict[str, Any]], int], None] = None,
                 species: SpeciesRetrivalDef = SpeciesRetrievalRepository.retrieve_species_1_gram) -> None:
        """
        Initializes the WindowEstimator class.

        :param completeness_threshold: The initial threshold for completeness metric.
        :param initial_data: Initial event data to populate the window.
        :param on_full_completeness: A callback function when full completeness is reached.
        :return: None.
        """
        if initial_data:
            self._window: List[Dict[str, Any]] = initial_data
        else:
            self._window: List[Dict[str, Any]] = []

        self.completeness_cache: List[List[float]] = []
        self.time_cache: List[tuple[int, float]] = []
        self.conformance_cache: List[tuple[int, ConformanceMetrics]] = []
        self.on_full_completeness = on_full_completeness
        self._window_counter: int = 0
        self.threshold_estimator: DynamicThresholdEstimator = DynamicThresholdEstimator()
        self.threshold_cache: List[float] = []

        self.species: SpeciesRetrivalDef = species

    def add_event(self, row_data: Dict[str, Any]) -> None:
        """
        Adds a new event to the current window.

        :param row_data: New event data to add to the window.
        :return: None.
        """
        self._window.append(row_data)

    def get_metrics(self, DEBUG: bool = False) -> float:
        """
        Calculates completeness metrics for the current window and updates caches. If the completeness
        exceeds the threshold, a callback is triggered.

        :param DEBUG: Flag to enable timing debug information.
        :return: The estimated completeness metric.
        """
        if DEBUG:
            start_time: float = time.time()

        estimated: float = self.calculate_coverage()

        # if len(self._window) <= 5:
        #     estimated = 0.0

        if DEBUG:
            self.time_cache.append((len(self._window), float(time.time() - start_time) * 1000))

        self.completeness_cache.append([len(self._window), estimated])

        # print(f"Completeness: {estimated:.2f}, Window Size: {len(self._window)}, {self.threshold_estimator.dynamic_threshold_heuristic():.2f}")

        dynamic_threshold: float = self.threshold_estimator.dynamic_threshold_heuristic([len(self._window), estimated])
        self.threshold_cache.append(dynamic_threshold)

        if (self.on_full_completeness is not None
                and estimated >= dynamic_threshold):
            self._window_counter += 1
            self.on_full_completeness(self._window, self._window_counter)
            self._window.clear()
            self.threshold_estimator.clear_cache()

        return estimated

    def get_window(self) -> List[Dict[str, Any]]:
        """
        Returns the current window of events.

        :return: The list of events in the window.
        """
        return self._window

    def calculate_coverage(self) -> float:
        """
        Calculates the coverage metric for the current window.
        The coverage metric is the ratio of the number of unique species (activities) in the current window
        :return:
        """
        estimator = SpeciesEstimator(step_size=None)
        estimator.register(self.species[0], self.species[1])

        estimator.apply(pd.DataFrame(self._window), verbose=False)

        reference_sample_incidence = estimator.metrics.get(self.species[0]).reference_sample_incidence
        incidence_current_total_species_count = estimator.metrics.get(
            self.species[0]).incidence_current_total_species_count

        coverage_incidence = coverage(reference_sample_incidence, incidence_current_total_species_count)

        return coverage_incidence
