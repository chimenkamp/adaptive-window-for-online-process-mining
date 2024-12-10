from abc import ABC
from collections import deque
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Callable, Deque, Dict, Tuple, Type, List, Optional
import math

from pm4py.objects.log.obj import EventLog, Event

class AbstractWindow(ABC):
    def __init__(self, on_window: Callable[[List[Dict[str, Any]], Optional[int]], None]) -> None:
        self.memory: Deque[Dict[str, Any]] = deque()
        self.on_window: Callable[[List[Dict[str, Any], Optional[int]]], None] = on_window

    def observe_event(self, event: Dict[str, Any], index: Optional[int]) -> None:
        pass

    def get_window(self) -> List[Dict[str, Any]]:
        pass


class SlidingWindow(AbstractWindow):
    def __init__(self, on_window: Callable[[List[Dict[str, Any]], Optional[int]], None]) -> None:
        super().__init__(on_window=on_window)


class TumblingWindow(AbstractWindow):
    def __init__(self, on_window: Callable[[List[Dict[str, Any]], Optional[int]], None]) -> None:
        super().__init__(on_window=on_window)


class AdaptiveWindow(AbstractWindow):
    def __init__(self, on_window: Callable[[List[Dict[str, Any]], Optional[int]], None]) -> None:
        super().__init__(on_window=on_window)


# Algorithm 1: Count-Based window model process mining algorithm
class CountBasedWindow(SlidingWindow):
    def __init__(self, max_memory: int, on_window: Callable[[List[Dict[str, Any]], Optional[int]], None]) -> None:
        super().__init__(on_window)
        self.max_m: int = max_memory

    def observe_event(self, event: Dict[str, Any], index: Optional[int]) -> None:
        if len(self.memory) >= self.max_m:
            self.on_window(self.get_window(), index)
            self.memory.popleft()

        self.memory.append(event)

    def get_window(self) -> List[Dict[str, Any]]:
        return list(self.memory)


# Algorithm 2: Lossy Counting window
class LossyCounting(TumblingWindow):
    def __init__(self, epsilon: float, on_window: Callable[[List[Dict[str, Any]], Optional[int]], None]) -> None:
        super().__init__(on_window)
        self.epsilon: float = epsilon
        self.T: Deque[Tuple[Dict[str, Any], Tuple[int, int]]] = deque()
        self.N: int = 1
        self.w: int = math.ceil(1 / epsilon)

    def observe_event(self, event: Dict[str, Any], index: Optional[int]) -> None:
        b_curr: int = math.ceil(self.N / self.w)

        found: bool = False
        for idx, (e, (f, d)) in enumerate(self.T):
            if e == event:
                self.T[idx] = (e, (f + 1, d))
                found = True
                break

        if not found:
            self.T.append((event, (1, b_curr - 1)))

        if self.N % self.w == 0:
            self.memory = self.get_window()
            self.on_window(self.memory, index)
            self.cleanup(b_curr)

        self.N += 1

    def cleanup(self, b_curr: int) -> None:
        self.T = deque([(e, (f, d)) for e, (f, d) in self.T if f + d > b_curr])

    def get_window(self) -> List[Dict[str, Any]]:
        window: Deque[Dict[str, Any]] = deque()
        for e, (f, d) in self.T:
            e_copy: Dict[str, Any] = e.copy()
            e_copy.update({'frequency': f, 'delta': d})
            window.append(e_copy)
        return list(window)


class LossyCountingWithBudget(LossyCounting):
    def __init__(self, epsilon: float, budget: int, on_window: Callable[[List[Dict[str, Any]], Optional[int]], None]) -> None:
        super().__init__(epsilon, on_window)
        self.budget: int = budget

    def observe_event(self, event: Dict[str, Any], index: Optional[int]) -> None:
        super().observe_event(event, index)
        if len(self.T) > self.budget:
            self.enforce_budget()

    def enforce_budget(self) -> None:
        self.T = deque(sorted(self.T, key=lambda x: x[1][0] + x[1][1]))
        while len(self.T) > self.budget:
            self.T.popleft()


class LandmarkWindow(TumblingWindow):
    def __init__(self, landmark: str, on_window: Callable[[List[Dict[str, Any]]], None]) -> None:
        super().__init__(on_window)
        self.landmark: str = landmark

    def observe_event(self, event: Dict[str, Any], index: Optional[int]) -> None:
        if self.landmark in event["concept:name"] and len(self.memory) > 1:
            self.on_window(self.get_window(), index)
            self.memory.clear()

        self.memory.append(event)

    def get_window(self) -> List[Dict[str, Any]]:
        return list(self.memory)


class TimeBasedSlidingWindow(SlidingWindow):
    def __init__(self, window_size: timedelta, on_window: Callable[[List[Dict[str, Any]]], None]) -> None:
        super().__init__(on_window)
        self.window_size_delta: timedelta = window_size

    def observe_event(self, event: Dict[str, Any], index: Optional[int]) -> None:
        current_time: datetime = event.get("time:timestamp")
        self.memory.append(event)
        self._remove_old_events(current_time)

    def get_window(self) -> List[Dict[str, Any]]:
        return list(self.memory)

    def _remove_old_events(self, current_time: datetime) -> None:
        while self.memory and (current_time - self.memory[0].get("time:timestamp")) > self.window_size_delta:
            self.on_window(self.get_window())
            self.memory.popleft()


class TimeBasedTumblingWindow(TumblingWindow):
    def __init__(self, window_size: timedelta, on_window: Callable[[List[Dict[str, Any]]], None]) -> None:
        super().__init__(on_window)
        self.window_size: timedelta = window_size
        self.window_start_time: Optional[datetime] = None

    def observe_event(self, event: Dict[str, Any], index: Optional[int] = None) -> None:
        event_time: datetime = event.get("time:timestamp").replace(tzinfo=None)

        if self.window_start_time is None:
            self.window_start_time = event_time

        if (event_time - self.window_start_time) >= self.window_size:
            self.on_window(self.get_window(), index)
            self.memory.clear()
            self.window_start_time = event_time

        self.memory.append(event)

    def get_window(self) -> List[Dict[str, Any]]:
        return list(self.memory)

