import os

import pandas as pd

from cna import process_cna_data
from rppa import process_rppa_data
from cancer_type import process_cancer_type_data
from tumor_purity import process_tumor_purity_data
from gex import process_gex_data
from entrezgene_id_to_mean_and_std_gex_mapping import process_entrezgene_id_to_mean_and_std_gex_mapping_data
from gene_predictability import process_gene_predictability_data

from utils import get_dfs_with_intersecting_sample_ids, get_dfs_with_intersecting_columns


data_dir = "../../data" # FIXME ("/cluster/scratch/aarslan/cna2gex_data")
raw_folder_name = "raw"
processed_folder_name = "processed"
os.makedirs(os.path.join(data_dir, processed_folder_name), exist_ok=True)
tumor_sample_ids = ["0" + str(i) for i in range(1, 10)]


if __name__ == "__main__":
    thresholded_cna_df, unthresholded_cna_df = process_cna_data(data_dir=data_dir, raw_folder_name=raw_folder_name, processed_folder_name=processed_folder_name)
    cancer_type_df = process_cancer_type_data(data_dir=data_dir, raw_folder_name=raw_folder_name, processed_folder_name=processed_folder_name)
    tumor_purity_df = process_tumor_purity_data(data_dir=data_dir, raw_folder_name=raw_folder_name, tumor_sample_ids=tumor_sample_ids)
    gex_df = process_gex_data(data_dir=data_dir, raw_folder_name=raw_folder_name, processed_folder_name=processed_folder_name, tumor_sample_ids=tumor_sample_ids)
    gex_df, unthresholded_cna_df, thresholded_cna_df, tumor_purity_df, cancer_type_df = get_dfs_with_intersecting_sample_ids(dfs=[gex_df, unthresholded_cna_df, thresholded_cna_df, tumor_purity_df, cancer_type_df])
    gex_df, unthresholded_cna_df, thresholded_cna_df = get_dfs_with_intersecting_columns(dfs=[gex_df, unthresholded_cna_df, thresholded_cna_df])

    rppa_df = process_rppa_data(data_dir=data_dir, raw_folder_name=raw_folder_name, processed_folder_name=processed_folder_name, intersecting_columns=gex_df.columns.tolist(), intersecting_sample_ids=gex_df["sample_id"].tolist())
    rppa_df.to_csv(os.path.join(data_dir, processed_folder_name, "rppa.tsv"), sep="\t", index=False)
    print("rppa_df.shape:", rppa_df.shape)

    rppa_genes_df = pd.DataFrame.from_dict({"gene_id": rppa_df.drop(columns=["sample_id"]).columns.tolist()})
    rppa_genes_df.to_csv(os.path.join(data_dir, processed_folder_name, "rppa_genes.tsv"), sep="\t", index=False)
    del rppa_df
    del rppa_genes_df

    thresholded_cna_df.to_csv(os.path.join(data_dir, processed_folder_name, "thresholded_cna.tsv"), sep="\t", index=False)
    print("thresholded_cna_df.shape:", thresholded_cna_df.shape)

    unthresholded_cna_df.to_csv(os.path.join(data_dir, processed_folder_name, "unthresholded_cna.tsv"), sep="\t", index=False)
    print("unthresholded_cna_df.shape:", unthresholded_cna_df.shape)
    del unthresholded_cna_df

    tumor_purity_df.to_csv(os.path.join(data_dir, processed_folder_name, "tumor_purity.tsv"), sep="\t", index=False)
    print("tumor_purity_df.shape:", tumor_purity_df.shape)
    del tumor_purity_df

    cancer_type_df.to_csv(os.path.join(data_dir, processed_folder_name, "cancer_type.tsv"), sep="\t", index=False)
    print("cancer_type_df.shape:", cancer_type_df.shape)

    cancer_type_one_hot_df = pd.get_dummies(data=cancer_type_df, columns=["cancer_type"])
    cancer_type_one_hot_df.to_csv(os.path.join(data_dir, processed_folder_name, "cancer_type_one_hot.tsv"), sep="\t", index=False)
    print("cancer_type_one_hot_df.shape", cancer_type_one_hot_df.shape)
    del cancer_type_df
    del cancer_type_one_hot_df

    gex_df.to_csv(os.path.join(data_dir, processed_folder_name, "gex.tsv"), sep="\t", index=False)
    print("gex_df.shape:", gex_df.shape)

    highly_expressed_genes_df = pd.DataFrame.from_dict({"gene_id": gex_df.drop(columns=["sample_id"]).columns.tolist(),
                                                        "gene_expression_sum": gex_df.drop(columns=["sample_id"]).sum(axis=0).values.ravel()})
    highly_expressed_genes_df = highly_expressed_genes_df.sort_values(by="gene_expression_sum", ascending=True).reset_index(drop=True)
    highly_expressed_genes_df.iloc[:1000, :][["gene_id"]].to_csv(os.path.join(data_dir, processed_folder_name, "1000_highly_expressed_genes.tsv"), sep="\t", index=False)
    highly_expressed_genes_df.iloc[:5000, :][["gene_id"]].to_csv(os.path.join(data_dir, processed_folder_name, "5000_highly_expressed_genes.tsv"), sep="\t", index=False)
    del highly_expressed_genes_df

    entrezgene_id_to_mean_and_std_gex_mapping_df = process_entrezgene_id_to_mean_and_std_gex_mapping_data(gex_df=gex_df, thresholded_cna_df=thresholded_cna_df)
    entrezgene_id_to_mean_and_std_gex_mapping_df.to_csv(os.path.join(data_dir, processed_folder_name, "entrezgene_id_to_mean_and_std_gex_mapping.tsv"), sep="\t", index=False)
    print("entrezgene_id_to_mean_and_std_gex_mapping_df.shape:", entrezgene_id_to_mean_and_std_gex_mapping_df.shape)

    gene_predictability_df = process_gene_predictability_data(gex_df=gex_df, entrezgene_id_to_mean_and_std_gex_mapping_df=entrezgene_id_to_mean_and_std_gex_mapping_df, thresholded_cna_df=thresholded_cna_df)
    gene_predictability_df.to_csv(os.path.join(data_dir, processed_folder_name, "entrezgene_id_to_aug_adg_ddg_dug_ratios_mapping.tsv"), sep="\t", index=False)
    print("gene_predictability_df.shape:", gene_predictability_df.shape)
