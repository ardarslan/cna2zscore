import os
from typing import List

import pandas as pd


def process_tumor_purity_data(data_dir: str, raw_folder_name: str, tumor_sample_ids: List[str]) -> pd.DataFrame:
    print("Processing Tumor Purity data...")

    tumor_purity_cpe_file_name = "tumor_purity.csv"
    tumor_purity_estimate_file_name = "tumor_purity_ESTIMATE.csv"

    tumor_purity_cpe_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, tumor_purity_cpe_file_name))
    tumor_purity_estimate_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, tumor_purity_estimate_file_name))

    tumor_sample_id_purity_mapping = {}
    for _, row in tumor_purity_cpe_df.iterrows():
        sample_id = row["Sample.ID"]
        purity_cpe = row["CPE"]
        purity_estimate = row["ESTIMATE"]
        purity_absolute = row["ABSOLUTE"]

        if not pd.isnull(purity_cpe):
            tumor_sample_id_purity_mapping[sample_id] = float(purity_cpe.replace(",", "."))
            continue

        if not pd.isnull(purity_absolute):
            tumor_sample_id_purity_mapping[sample_id] = float(purity_absolute.replace(",", "."))
            continue

        if not pd.isnull(purity_estimate):
            tumor_sample_id_purity_mapping[sample_id] = float(purity_estimate.replace(",", "."))
            continue

    for _, row in tumor_purity_estimate_df.iterrows():
        sample_id = row["NAME"]
        purity_estimate = row["TumorPurity"]
        if not pd.isnull(purity_estimate):
            tumor_sample_id_purity_mapping[sample_id] = purity_estimate

    tumor_purity_df = pd.DataFrame.from_dict({"sample_id": tumor_sample_id_purity_mapping.keys(),
                                              "purity": tumor_sample_id_purity_mapping.values()})

    sample_id_dict = {}
    for sample_id in tumor_purity_df["sample_id"].values:
        if sample_id.split("-")[3][:2] not in tumor_sample_ids:
            continue
        sample_id_first_15 = sample_id[:15]
        if sample_id_first_15 in sample_id_dict.keys():
            if sample_id < sample_id_dict[sample_id_first_15]:
                sample_id_dict[sample_id_first_15] = sample_id
            else:
                continue
        else:
            sample_id_dict[sample_id_first_15] = sample_id

    tumor_purity_df = tumor_purity_df[tumor_purity_df["sample_id"].swifter.apply(lambda x: x in list(sample_id_dict.values()))]
    tumor_purity_df["sample_id"] = tumor_purity_df["sample_id"].swifter.apply(lambda x: x[:15])

    print("Processed Tumor Purity data.")

    return tumor_purity_df
