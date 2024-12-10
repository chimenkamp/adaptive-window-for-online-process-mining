import os
from random import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import pm4py


def create_sudden_drift(logs: List[pd.DataFrame]) -> pd.DataFrame:
    """Create a sudden drift by concatenating two logs at a specific point."""
    return pd.concat(logs, ignore_index=True)


def create_recurring_drift(logs: List[pd.DataFrame]) -> pd.DataFrame:
    """Create a recurring drift by alternating between the logs in a cycle."""
    if len(logs) < 2:
        raise ValueError("At least two logs are required.")

    # Split the first log in two parts correctly
    log1 = logs[0]
    split_point = len(log1) // 2
    log1_part1 = log1.iloc[:split_point]
    log1_part2 = log1.iloc[split_point:]

    # Combine logs
    combined_log = pd.concat([log1_part1, logs[1], log1_part2], ignore_index=True)

    return combined_log


def generate_boolean_list(n: int) -> List[bool]:
    def probability_function(x: float) -> float:
        return 0.5 + 0.3 * (1 - np.exp(-5 * x))

    boolean_list = []
    for i in range(n):
        probability = probability_function(i / n)
        boolean_value = random() < probability
        boolean_list.append(boolean_value)

    return boolean_list


def create_gradual_drift(log_a: pd.DataFrame, log_b: pd.DataFrame, drift_start: int, drift_end: int) -> pd.DataFrame:
    """Create a gradual drift by linearly combining
    the two logs between drift_start and drift_end with a probability of replacement."""

    columns: List[str] = ["case:concept:name", "time:timestamp", "concept:name"]
    log_a = log_a[columns]
    log_b = log_b[columns]

    if drift_start < 0 or drift_end > len(log_a) or drift_start >= drift_end:
        raise ValueError("Invalid drift_start or drift_end values.")

    before_drift = log_a.iloc[:drift_start]
    drift_segment = log_a.iloc[drift_start:drift_end].reset_index(drop=True)
    combination_function = generate_boolean_list(len(drift_segment))

    for i in range(len(drift_segment)):
        if i < len(log_b) and combination_function[i]:
            row_to_insert = log_b.iloc[i]
            drift_segment = pd.concat(
                [
                    drift_segment.iloc[:i],
                    pd.DataFrame([row_to_insert]),
                    drift_segment.iloc[i + 1:]
                ]
            ).reset_index(drop=True)

    after_drift = log_b.reset_index(drop=True)
    merged_df = pd.concat([before_drift, drift_segment, after_drift]).reset_index(drop=True)

    return merged_df


def create_incremental_drift(log_a: pd.DataFrame, log_b: pd.DataFrame,
                             increment_logs: list[pd.DataFrame]) -> pd.DataFrame:
    """Create an incremental drift by gradually increasing the proportion of the second log in the combined log."""

    columns: List[str] = ["case:concept:name", "time:timestamp", "concept:name"]
    log_a: pd.DataFrame = log_a[columns]
    log_b: pd.DataFrame = log_b[columns]
    increment_logs: list[pd.DataFrame] = [log[columns].head(50) for log in increment_logs]

    all_logs: List[pd.DataFrame] = [log_a]

    for log in increment_logs:
        all_logs.append(log)

    all_logs.append(log_b)

    # Combine the logs
    combined_log = pd.concat(all_logs, ignore_index=True)

    return combined_log


def filter_first_n_cases(df: pd.DataFrame, n: int) -> pd.DataFrame:
    unique_case_ids = df['case:concept:name'].unique()[:n]
    filtered_df = df[df['case:concept:name'].isin(unique_case_ids)]
    return filtered_df


def average_case(df: pd.DataFrame) -> float:
    case_lengths = df.groupby('case:concept:name').size()
    average_length = case_lengths.mean()
    return average_length


DATA_PATH: str = "data/"
LOG_NAME_A: str = "sepsis/Sepsis Cases - Event Log.feather"
# LOG_NAME: str = "Community/daily_living/activity_log.xes"
# LOG_NAME: str = "Synthetic/synthetic, online order/online_order.xes"
LOGS = [
    ("Community/bpi-c-2012/data.feather", 10),
    ("Community/bpi-c-2013/data.feather", 50),
    ("Community/bpi-c-2015/data.feather", 4),
    ("Community/bpi-c-2018/data.feather", 4),
    ("Community/bpi-c-2017/data.feather", 4),
    ("Community/bpi-c-2019/data.feather", 4),
]

dataframes = [filter_first_n_cases(pd.read_feather(DATA_PATH + log[0]), log[1]) for log in LOGS]

s = 0
for i, df in enumerate(dataframes):
    s += len(df)
    print(LOGS[i][0].split("/")[1] + "\t", str(average_case(df)) + "\t", str(len(df)) + "\t", str(s) + "\t")

# sudden_drift = create_sudden_drift(dataframes)
# recurring_drift = create_recurring_drift(dataframes)
# gradual_drift = create_gradual_drift(dataframes[0], dataframes[1], drift_start=75, drift_end=160)
incremental_drift = create_incremental_drift(dataframes[0], dataframes[1], dataframes[2:])

SAVE_PATH: str = "data/Concept_Drift/Merged"
incremental_drift.to_feather(os.path.join(SAVE_PATH, "incremental_drift_short[bpi-c-2012,bpi-c-2013].feather"))
incremental_drift.to_csv(os.path.join(SAVE_PATH, "incremental_drift_short[bpi-c-2012,bpi-c-2013,bpi-c-2015,bpi-c-2017,bpi-c-2019].feather].csv"),
                         index=False)
