import os
from typing import List

import numpy as np
import pandas as pd


def process_rppa_data(data_dir: str, raw_folder_name: str, processed_folder_name: str, intersecting_columns: List[str], intersecting_sample_ids: List[str]) -> pd.DataFrame:
    print("Processing RPPA data...")

    rppa_file_name = "TCGA-RPPA-pancan-clean.xena"
    protein_name_to_hgnc_symbol_mapping_file_name = "tcpa_to_ncbi_mapping.csv"
    hgnc_symbol_to_entrezgene_id_mapping_file_name = "hgnc_to_entrezgene_id_mapping.tsv"

    rppa_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, rppa_file_name), sep="\t")
    hgnc_symbol_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, processed_folder_name, hgnc_symbol_to_entrezgene_id_mapping_file_name), sep="\t").values)
    protein_name_to_hgnc_symbol_mapping_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, protein_name_to_hgnc_symbol_mapping_file_name), sep=",")
    protein_name_to_hgnc_symbol_mapping = dict(protein_name_to_hgnc_symbol_mapping_df[["TCPA Symbol", "NCBI Symbol 1"]].values)

    rppa_df["SampleID"] = rppa_df["SampleID"].apply(lambda x: hgnc_symbol_to_entrezgene_id_mapping.get(protein_name_to_hgnc_symbol_mapping[x], -1))
    rppa_df = rppa_df[rppa_df["SampleID"] != -1]
    rppa_df = rppa_df.dropna(axis=0)

    def select_gene_with_max_total_expression(x):
        gene_total_expressions = x.drop(columns=["SampleID"]).values.sum(axis=1).ravel()
        max_expression_index = np.argmax(gene_total_expressions)
        return x.iloc[max_expression_index, :]

    rppa_df = rppa_df.groupby("SampleID").apply(lambda x: select_gene_with_max_total_expression(x)).reset_index(drop=True)

    rppa_df.index = rppa_df["SampleID"].tolist()
    rppa_df.drop(columns=["SampleID"], inplace=True)
    rppa_df = rppa_df.T
    rppa_df.reset_index(drop=False, inplace=True)
    rppa_df.rename(columns={"index": "sample_id"}, inplace=True)
    rppa_df.columns = ["sample_id"] + [int(column) for column in rppa_df.columns if column != "sample_id"]

    print("Processed RPPA data.")

    return rppa_df
