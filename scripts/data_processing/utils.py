from typing import List

import pandas as pd


def get_dfs_with_intersecting_sample_ids(dfs: List[pd.DataFrame]) -> List[str]:
    intersecting_sample_ids = None
    for df in dfs:
        if intersecting_sample_ids is None:
            intersecting_sample_ids = set(df["sample_id"].tolist())
        else:
            intersecting_sample_ids = intersecting_sample_ids.intersection(df["sample_id"].tolist())

    for index in range(dfs):
        dfs[index] = df[df["sample_id"].isin(intersecting_sample_ids)].sort_values(by="sample_id")

    return dfs


def get_dfs_with_intersecting_columns(dfs: List[pd.DataFrame]) -> List[str]:
    intersecting_columns = None
    for df in dfs:
        if intersecting_columns is None:
            intersecting_columns = set(df.drop(columns=["sample_id"]).columns)
        else:
            intersecting_columns = intersecting_columns.intersection(set(df.drop(columns=["sample_id"]).columns))

    for index in range(dfs):
        dfs[index] = df[["sample_id"] + sorted(list(intersecting_columns))]

    return dfs
