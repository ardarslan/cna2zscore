import os
import numpy as np
import pandas as pd


def get_breast_cancer_scc_genes_data(data_dir, raw_folder_name, processed_folder_name):
    breast_cancer_scc_genes_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, "breast_cancer_scc_genes.tsv"), sep="\t")
    hgnc_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, processed_folder_name, "hgnc_to_entrezgene_id_mapping.tsv"), sep="\t").values)
    breast_cancer_scc_genes_df["gene_id"] = breast_cancer_scc_genes_df["hgnc_symbol"].apply(lambda x: hgnc_to_entrezgene_id_mapping.get(x, np.NaN))
    breast_cancer_scc_genes_df.dropna(axis=1, inplace=True)
    return breast_cancer_scc_genes_df
