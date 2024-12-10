from typing import List, Dict, Callable, Any

from src.utils.methods import ConformanceMetrics


class DynamicThresholdEstimator:
    def __init__(
            self,
            base_threshold: float = 0.6,
            smoothing_factor: float = 0.2,
            decay_rate: float = 0.1,
            minimum_threshold: float = 0.01
    ) -> None:
        """
        Initializes the DynamicThresholdEstimator class with parameters for threshold updates.

        :param completeness_cache: A cache containing completeness metrics for each window.
        :param base_threshold: The initial static threshold value before dynamic adjustment.
        :param smoothing_factor: A factor to control how much the threshold can change between windows.
        :param decay_rate: The rate at which the threshold should decay when completeness remains low.
        :param minimum_threshold: The minimum value the threshold can decrease to.
        :return: None.
        """
        self.completeness_cache: List[List[float]] = []
        self.base_threshold = base_threshold
        self.smoothing_factor = smoothing_factor
        self.current_threshold = base_threshold
        self.decay_rate = decay_rate
        self.minimum_threshold = minimum_threshold  # Initialize minimum_threshold

    def dynamic_threshold_heuristic(self, completeness: List[float]) -> float:
        """
        Implements the heuristic to dynamically adjust the threshold, with smoothing and decay applied.

        :return: The dynamic threshold value.
        """

        self.completeness_cache.append(completeness)

        if len(self.completeness_cache) < 3:
            return self.current_threshold  # Not enough data to apply the heuristic.

        # Calculate the second derivative of completeness over time.
        r_prime_2: list[float] = self.calculate_second_derivative()

        # Identify the point with the highest change in the second derivative (the "elbow").
        optimal_threshold_idx = max(range(len(r_prime_2)), key=lambda i: r_prime_2[i])
        optimal_threshold_value = self.completeness_cache[optimal_threshold_idx + 1][1]

        # Dynamically adjust the smoothing factor if stagnation is detected.
        if self.is_completeness_stagnating():
            self.smoothing_factor = min(self.smoothing_factor * 1.2, 0.99)  # Increase smoothing if stagnating
            # Allow threshold to decrease below base_threshold down to minimum_threshold
            self.current_threshold = max(self.current_threshold - self.decay_rate, self.minimum_threshold)
        else:
            self.smoothing_factor = max(self.smoothing_factor * 0.8,
                                        0.01)  # Decrease smoothing to react faster if not stagnating

        # Smooth the threshold update to avoid abrupt changes.
        self.current_threshold = (
                self.smoothing_factor * optimal_threshold_value
                + (1 - self.smoothing_factor) * self.current_threshold
        )

        return self.current_threshold

    def calculate_second_derivative(self) -> List[float]:
        """
        Calculates the second-order difference of the completeness metric over time.

        :return: List of second-order differences.
        """
        r_prime_2: List[float] = []
        for i in range(1, len(self.completeness_cache) - 1):
            r_prime_2_value = (
                    self.completeness_cache[i - 1][1]
                    - 2 * self.completeness_cache[i][1]
                    + self.completeness_cache[i + 1][1]
            )
            r_prime_2.append(r_prime_2_value)
        return r_prime_2

    def is_completeness_stagnating(
            self, stagnation_threshold: float = 0.1, stagnation_window: int = 10
    ) -> bool:
        """
        Checks if the completeness metric has stagnated for a number of consecutive windows.

        :param stagnation_threshold: A threshold for detecting whether completeness is stagnating.
        :param stagnation_window: The number of consecutive windows to check for stagnation.
        :return: True if completeness has stagnated, otherwise False.
        """
        if len(self.completeness_cache) < stagnation_window:
            return False

        recent_completeness = [
            self.completeness_cache[-i][1] for i in range(1, stagnation_window + 1)
        ]

        return all(
            abs(recent_completeness[i] - recent_completeness[i + 1]) < stagnation_threshold
            for i in range(stagnation_window - 1)
        )

    def clear_cache(self) -> None:
        """
        Clears the completeness cache.

        :return: None.
        """
        self.completeness_cache.clear()