import os
import numpy as np
import pandas as pd
from typing import Dict


def get_breast_cancer_scc_genes_data(data_dir: str, raw_folder_name: str, processed_folder_name: str) -> Dict[str, pd.DataFrame]:
    cna_components = pd.read_excel(os.path.join(data_dir, raw_folder_name, "journal.pone.0276886.s002.xlsx"), sheet_name="Supplementary Table 1")
    cna_components = cna_components.drop(columns=["Unnamed: 0"])
    zscore_components = pd.read_excel(os.path.join(data_dir, raw_folder_name, "journal.pone.0276886.s002.xlsx"), sheet_name="Supplementary Table 2")
    zscore_components = zscore_components.drop(columns=["Unnamed: 0"])

    hgnc_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, processed_folder_name, "hgnc_to_entrezgene_id_mapping.tsv"), sep="\t").values)

    breast_cancer_scc_genes_df_mapping = {}
    for column in cna_components.columns:
        cna_components[column] = cna_components[column].apply(lambda x: hgnc_to_entrezgene_id_mapping.get(x, np.NaN))
        zscore_components[column] = zscore_components[column].apply(lambda x: hgnc_to_entrezgene_id_mapping.get(x, np.NaN))
        breast_cancer_scc_genes_df_mapping[str(column)] = pd.DataFrame.from_dict({"gene_id": cna_components[column].dropna(axis=0).apply(int).tolist() + zscore_components[column].dropna(axis=0).apply(int).tolist()})

    return breast_cancer_scc_genes_df_mapping
