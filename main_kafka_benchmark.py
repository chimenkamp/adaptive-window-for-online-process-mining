import asyncio
import signal
import warnings
from datetime import timedelta, datetime
from typing import List, Dict, Any

from src.utils.windowing_baseline import (
    CountBasedWindow,
    LossyCounting,
    LossyCountingWithBudget,
    LandmarkWindow,
    TimeBasedSlidingWindow,
    TimeBasedTumblingWindow,
    AbstractWindow,
    EstimatorWindow
)

from faust import App, Stream
import pandas as pd
from time import time

from src.utils.methods import PerformanceMetrics

warnings.filterwarnings('ignore')

# Execution Constants
DATA_PATH: str = "data/"
LOG_NAME: str = "sepsis/Sepsis Cases - Event Log"
DATASET: pd.DataFrame = pd.read_feather(DATA_PATH + LOG_NAME + ".feather")
EVALUATION_TIME_IN_SECONDS: int = 60*3

# App Initialization
app: App = App('WindowingApp', broker='kafka://localhost:9092', value_serializer='json')
topic = app.topic('completenessEstimation')

window_storage: List[Dict[str, Any]] = []

window_counter: int = 0

performance_metrics: PerformanceMetrics = PerformanceMetrics()


def on_window(w: List[Dict[str, Any]], index: int) -> None:
    global window_counter
    window_counter += 1
    performance_metrics.calculate_step(len_window=100, window_counter=window_counter)


# window: AbstractWindow = CountBasedWindow(100, on_window=on_window)


# For LossyCounting with epsilon = 0.01
# window: AbstractWindow = LossyCounting(0.05, on_window=on_window)


# For LossyCountingWithBudget with epsilon = 0.01 and budget = 500
# window: AbstractWindow = LossyCountingWithBudget(0.01, 10, on_window=on_window)

# For LandmarkWindow with landmark = "landmark_event"
# window: AbstractWindow = LandmarkWindow("ER Registration", on_window=on_window)

# For TimeBasedSlidingWindow with window_size = timedelta(conds=5)
# window: AbstractWindow = TimeBasedSlidingWindow(timedelta(seconds=5), on_window=on_window)

# For TimeBasedTumblingWindow with window_size = timedelta(seconds=5)
# window: AbstractWindow = TimeBasedTumblingWindow(timedelta(seconds=5), on_window=on_window)


# For EstimatorWindow with completeness_threshold = 0.90
window: AbstractWindow = EstimatorWindow(0.90, on_window=on_window)


# Count-based window
@app.agent(topic)
async def process(stream: Stream, ) -> None:
    print("Processing stream...")

    track_time: float = time()
    performance_metrics.reset()
    event_count: int = 0
    async for event in stream:
        event['time:timestamp'] = datetime.now()
        if time() - track_time > EVALUATION_TIME_IN_SECONDS:
            break
        performance_metrics.processed_events += 1
        event_count += 1
        window.observe_event(event, 0)

    performance_metrics.save("ESTIMATOR_COMP_METRICS")


@app.timer(interval=15.0)
async def send_rows() -> None:
    batch_size: int = 7500  # Maintain the optimized batch size
    tasks: List[asyncio.Task] = []
    for row in DATASET.itertuples(index=False, name=None):
        row_data: Dict[str, Any] = {col: value for col, value in zip(DATASET.columns, row)}

        task: asyncio.Task = topic.send(value=row_data)
        tasks.append(task)

        if len(tasks) >= batch_size:
            await asyncio.gather(*tasks)
            tasks = []

    if tasks:
        await asyncio.gather(*tasks)


async def shutdown() -> None:
    print("Shutting down gracefully...")
    await app.stop()


def handle_signal(signal_number: int, frame: Any) -> None:
    asyncio.create_task(shutdown())


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    app.main()

# BASELINE CODE
# @app.agent(topic)
# async def process(stream: Stream) -> None:
#     print("Processing stream...")
#     performance_metrics.reset()
#     window_counter: int = 0
#
#     track_time: float = time()
#
#     async for event in stream:
#         if time() - track_time > EVALUATION_TIME_IN_SECONDS:
#             break
#
#         performance_metrics.processed_events += 1
#
#         if performance_metrics.processed_events % 100 == 0:
#             window_counter += 1
#             performance_metrics.calculate_step(len_window=100, window_counter=window_counter)
#
#     performance_metrics.save("BASELINE_COMP_METRICS")
