import numpy as np
import pandas as pd


def get_zscore_data(gex_df: pd.DataFrame, cancer_type_df: pd.DataFrame):
    gex_column_names = gex_df.drop(columns=["sample_id"]).columns.tolist()
    merged_df = pd.merge(gex_df, cancer_type_df, how="inner", on="sample_id")
    merged_df = merged_df[["sample_id"] + [column for column in merged_df.columns if column != "sample_id"]]

    def calculate_zscores(x):
        gex = x.values[:, 1:x.shape[1]-1].astype(np.float32)
        gex_mean = np.mean(gex, axis=0)
        gex_std = np.std(gex, axis=0) + 1e-10
        return pd.concat([x[["sample_id"]].reset_index(drop=True), pd.DataFrame(data=(gex-gex_mean)/gex_std, columns=gex_column_names)], axis=1)

    zscore_df = merged_df.groupby("cancer_type").apply(lambda x: calculate_zscores(x)).reset_index(drop=True)
    zscore_df = pd.merge(gex_df[["sample_id"]], zscore_df, how="left", on="sample_id")
    return zscore_df
