import os
from typing import List

import numpy as np
import pandas as pd


def process_gex_data(data_dir: str, processed_folder_name: str, tumor_sample_ids: List[str]) -> pd.DataFrame:
    print("Processing GEX data...")

    gex_file_name = "tcga_RSEM_gene_tpm"

    ensembl_id_to_entrezgene_id_mapping_file_name = "ensembl_id_to_entrezgene_id_mapping.tsv"

    ensembl_id_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, processed_folder_name, ensembl_id_to_entrezgene_id_mapping_file_name), sep="\t").values)

    gex_df = pd.read_csv(os.path.join(data_dir, "raw", gex_file_name), sep="\t")

    gex_df.rename(columns={"sample": "ensembl_id"}, inplace=True)
    gex_df = gex_df[["ensembl_id"] + [column for column in gex_df.columns if column.split("-")[-1] in tumor_sample_ids]]
    gex_df["ensembl_id"] = gex_df["ensembl_id"].swifter.apply(lambda x: x.split(".")[0]).tolist()
    gex_df = gex_df[gex_df["ensembl_id"].swifter.apply(lambda x: x in ensembl_id_to_entrezgene_id_mapping.keys())]
    gex_df["entrezgene_id"] = gex_df["ensembl_id"].swifter.apply(lambda x: ensembl_id_to_entrezgene_id_mapping[x]).tolist()
    gex_df.drop(columns=["ensembl_id"], inplace=True)

    def get_indices_with_max_expression_per_gene(x):
        expression_sums = x["expression_sum"].values
        expression_argmax = np.argmax(expression_sums)
        return int(x.iloc[expression_argmax, :]["index"])

    gex_df.reset_index(drop=True, inplace=True)
    gex_df["index"] = gex_df.index.tolist()
    gex_df["expression_sum"] = gex_df.drop(columns=["entrezgene_id"]).values.sum(axis=1).tolist()
    selected_indices = gex_df[["index", "entrezgene_id", "expression_sum"]].swifter.groupby("entrezgene_id").apply(lambda x: get_indices_with_max_expression_per_gene(x)).tolist()
    gex_df = gex_df[gex_df["index"].isin(selected_indices)].drop(columns=["index", "expression_sum"])

    gex_df["entrezgene_id"] = gex_df["entrezgene_id"].swifter.apply(lambda x: int(x))

    gex_df.index = gex_df["entrezgene_id"].tolist()
    gex_df.drop(columns=["entrezgene_id"], inplace=True)

    gex_df = gex_df.T
    gex_df.reset_index(drop=False, inplace=True)
    gex_df.rename(columns={"index": "sample_id"}, inplace=True)

    print("Processed GEX data.")

    return gex_df
