import dataclasses
from functools import partial

import numpy as np
import pandas as pd
import pm4py
import time
from typing import Optional, Union, List, Callable, TypeVar, Dict, Any, Tuple, Generic
from special4pm.estimation import SpeciesEstimator
from special4pm.estimation.metrics import coverage
from special4pm.species import (
    retrieve_species_n_gram,
    retrieve_species_trace_variant,
    retrieve_timed_activity,
    retrieve_timed_activity_exponential
)

from src.species_definition import SpeciesRetrievalRepository, SpeciesRetrivalDef
from src.utils.dynamic_threshold_estimator import DynamicThresholdEstimator
from src.utils.methods import ConformanceMetrics, calculate_conformance_metrics
import warnings
import time
from numpy.typing import NDArray


warnings.filterwarnings('ignore')


T = TypeVar('T')


class WindowEstimatorDebug(Generic[T]):
    def __init__(self,
                 initial_data: Optional[List[T]] = None,
                 on_full_completeness: Optional[Callable[[NDArray[T], int], None]] = None,
                 species: SpeciesRetrivalDef = SpeciesRetrievalRepository.retrieve_species_1_gram) -> None:
        """
        Adaptive window for streaming process mining. The window estimator is used to estimate the completeness of
        a stream of events. When the completeness exceeds a threshold, a callback function is triggered.

        Debug version of the WindowEstimator class with debug code and caching of metrics.

        :param initial_data: Initial event data to populate the window.
        :param on_full_completeness: A callback function when full completeness is reached.
        :param species: The species retrieval definition for coverage calculation.
        :return: None.
        """

        if initial_data:
            self._window: NDArray[T] = np.array(initial_data, dtype=object)
        else:
            self._window: NDArray[T] = np.array([], dtype=object)

        self.completeness_cache: List[List[float]] = []
        self.time_cache: List[Tuple[int, float]] = []
        self.conformance_cache: List[Tuple[int, ConformanceMetrics]] = []
        self.on_full_completeness = on_full_completeness
        self._window_counter: int = 0
        self.threshold_estimator: DynamicThresholdEstimator = DynamicThresholdEstimator()
        self.threshold_cache: List[float] = []

        self.species: SpeciesRetrivalDef = species

    def add_event(self, row_data: T) -> None:
        """
        Adds a new event to the current window.

        :param row_data: New event data to add to the window.
        :return: None.
        """
        self._window = np.append(self._window, np.array([row_data], dtype=object))

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

        if DEBUG:
            self.time_cache.append((len(self._window), float(time.time() - start_time) * 1000))

        self.completeness_cache.append([len(self._window), estimated])

        dynamic_threshold: float = self.threshold_estimator.dynamic_threshold_heuristic([len(self._window), estimated])
        self.threshold_cache.append(dynamic_threshold)

        if (self.on_full_completeness is not None
                and estimated >= dynamic_threshold):
            self._window_counter += 1
            self.on_full_completeness(self._window, self._window_counter)
            self._window = np.array([], dtype=object)  # Clear the window
            self.threshold_estimator.clear_cache()

        return estimated

    def get_window(self) -> NDArray[T]:
        """
        Returns the current window of events.

        :return: The NumPy array of events in the window.
        """
        return self._window

    def calculate_coverage(self) -> float:
        """
        Calculates the coverage metric for the current window.
        The coverage metric is the ratio of the number of unique species (activities) in the current window
        :return: The coverage metric as a float.
        """
        estimator = SpeciesEstimator(step_size=None)
        estimator.register(self.species[0], self.species[1])

        # Convert numpy array to DataFrame for estimator
        window_df = pd.DataFrame(self._window.tolist())

        # If window is empty, return 0 coverage
        if len(self._window) == 0:
            return 0.0

        estimator.apply(window_df, verbose=False)

        reference_sample_incidence = estimator.metrics.get(self.species[0]).reference_sample_incidence
        incidence_current_total_species_count = estimator.metrics.get(
            self.species[0]).incidence_current_total_species_count

        coverage_incidence = coverage(reference_sample_incidence, incidence_current_total_species_count)

        return coverage_incidence
