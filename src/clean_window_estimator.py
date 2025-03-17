from typing import List, Dict, Any, TypeVar, Generic, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from special4pm.estimation import SpeciesEstimator
from special4pm.estimation.metrics import coverage

from src.species_definition import SpeciesRetrivalDef, SpeciesRetrievalRepository
from src.utils.dynamic_threshold_estimator import DynamicThresholdEstimator

T = TypeVar('T')

class WindowEstimatorClean(Generic[T]):
    def __init__(self,
                 initial_data: Optional[List[T]] = None,
                 on_full_completeness: Optional[Callable[[NDArray[T], int], None]] = None,
                 species: SpeciesRetrivalDef = SpeciesRetrievalRepository.retrieve_species_1_gram) -> None:
        """
        Adaptive window for streaming process mining. The window estimator is used to estimate the completeness of
        a stream of events. When the completeness exceeds a threshold, a callback function is triggered.

        Clean version of the WindowEstimator class from the src/window_estimator.py file without debug code and caching of metrics.

        :param initial_data: Initial event data to populate the window.
        :param on_full_completeness: A callback function when full completeness is reached.
        :param species: The species retrieval definition for coverage calculation.
        :return: None.
        """

        if initial_data:
            self._window: NDArray[T] = np.array(initial_data, dtype=object)
        else:
            self._window: NDArray[T] = np.array([], dtype=object)

        self.on_full_completeness = on_full_completeness
        self._window_counter: int = 0
        self.threshold_estimator: DynamicThresholdEstimator = DynamicThresholdEstimator()
        self.species: SpeciesRetrivalDef = species

    def add_event(self, row_data: T) -> None:
        """
        Adds a new event to the current window.

        :param row_data: New event data to add to the window.
        :return: None.
        """
        self._window = np.append(self._window, np.array([row_data], dtype=object))

    def get_metrics(self) -> float:
        """
        Calculates completeness metrics for the current window. If the completeness
        exceeds the threshold, a callback is triggered.

        :return: The estimated completeness metric.
        """
        estimated: float = self.calculate_coverage()

        dynamic_threshold: float = self.threshold_estimator.dynamic_threshold_heuristic([len(self._window), estimated])

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
        # If window is empty, return 0 coverage
        if len(self._window) == 0:
            return 0.0

        estimator = SpeciesEstimator(step_size=None)
        estimator.register(self.species[0], self.species[1])

        # Convert numpy array to DataFrame for estimator
        window_df = pd.DataFrame(self._window.tolist())

        estimator.apply(window_df, verbose=False)

        reference_sample_incidence = estimator.metrics.get(self.species[0]).reference_sample_incidence
        incidence_current_total_species_count = estimator.metrics.get(
            self.species[0]).incidence_current_total_species_count

        coverage_incidence = coverage(reference_sample_incidence, incidence_current_total_species_count)

        return coverage_incidence