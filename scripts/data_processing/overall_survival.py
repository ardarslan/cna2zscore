import os
import pandas as pd


def get_overall_survival_data(data_dir: str, raw_folder_name: str) -> pd.DataFrame:
    overall_survival_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, "Survival_SupplementalTable_S1_20171025_xena_sp"), sep="\t")
    overall_survival_df.rename(columns={"sample": "sample_id", "OS": "overall_survival", "OS.time": "overall_survival_time"}, inplace=True)
    overall_survival_df = overall_survival_df[["sample_id", "overall_survival_time", "overall_survival"]]
    overall_survival_df = overall_survival_df.dropna(axis=0)
    overall_survival_df["overall_survival_time"] = overall_survival_df["overall_survival_time"].apply(lambda x: int(x))
    overall_survival_df["overall_survival"] = overall_survival_df["overall_survival"].apply(lambda x: int(x))
    overall_survival_df.drop_duplicates(subset=["sample_id"], inplace=True, keep="first")
    return overall_survival_df
