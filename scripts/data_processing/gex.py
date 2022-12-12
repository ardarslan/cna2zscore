import os
from typing import List

import numpy as np
import pandas as pd


def process_gex_data(data_dir: str, raw_folder_name: str, processed_folder_name: str, tumor_sample_ids: List[str]) -> pd.DataFrame:
    print("Processing GEX data...")

    gex_file_name = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
    hgnc_to_entrezgene_id_mapping_file_name = "hgnc_to_entrezgene_id_mapping.tsv"

    gex_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, gex_file_name), sep="\t")
    hgnc_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, processed_folder_name, hgnc_to_entrezgene_id_mapping_file_name), sep="\t").values)

    gex_df = gex_df.rename(columns={"sample": "entrezgene_id"})

    for index, row in gex_df.iterrows():
        if row["entrezgene_id"].isdigit():
            gex_df.at[index, "entrezgene_id"] = int(row["entrezgene_id"])
        else:
            gex_df.at[index, "entrezgene_id"] = hgnc_to_entrezgene_id_mapping.get(row["entrezgene_id"], np.NaN)

    gex_df = gex_df[~pd.isnull(gex_df["entrezgene_id"])]

    gex_df = gex_df.dropna(axis=0)

    gex_df = gex_df[["entrezgene_id"] + [column for column in gex_df.columns if column.split("-")[-1] in tumor_sample_ids]]

    def select_gene_with_max_total_expression(x):
        gene_total_expressions = x.drop(columns=["entrezgene_id"]).values.sum(axis=1).ravel()
        max_expression_index = np.argmax(gene_total_expressions)
        return x.drop(columns=["entrezgene_id"]).iloc[max_expression_index, :]

    gex_df = gex_df.groupby("entrezgene_id").apply(lambda x: select_gene_with_max_total_expression(x)).reset_index(drop=False)

    gex_df.set_index("entrezgene_id", inplace=True)

    gex_df = gex_df.T
    gex_df = gex_df.reset_index(drop=False)

    gex_df = gex_df.rename(columns={"index": "sample_id"})
    gex_df.rename_axis(None, axis=1, inplace=True)

    print("Processed GEX data.")

    return gex_df
