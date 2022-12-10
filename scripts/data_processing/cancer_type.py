import os
import pandas as pd


def process_cancer_type_data(data_dir: str, raw_folder_name: str, processed_folder_name: str) -> pd.DataFrame:
    print("Processing Cancer Type data...")

    cancer_type_file_name = "TCGA_phenotype_denseDataOnlyDownload.tsv"
    cancer_type_full_name_to_abbreviation_mapping_file_name = "cancer_type_full_name_to_abbreviation_mapping.tsv"

    cancer_type_full_name_to_abbreviation_mapping = dict(pd.read_csv(os.path.join(data_dir, processed_folder_name, cancer_type_full_name_to_abbreviation_mapping_file_name), sep="\t").values)

    cancer_type_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, cancer_type_file_name), sep="\t")
    cancer_type_df = cancer_type_df.rename(columns={"sample": "sample_id", "_primary_disease": "cancer_type"})
    cancer_type_df = cancer_type_df[["sample_id", "cancer_type"]]
    cancer_type_df["cancer_type"] = cancer_type_df["cancer_type"].swifter.apply(lambda x: cancer_type_full_name_to_abbreviation_mapping[x])

    print("Processed Cancer Type data...")

    return cancer_type_df
