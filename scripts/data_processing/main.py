import os

import pandas as pd
import ray
ray.init(ignore_reinit_error=True, num_cpus=16, memory=int(30 * 1e9))
import swifter

from cna import process_cna_data
from rppa import process_rppa_data
from cancer_type import process_cancer_type_data
from tumor_purity import process_tumor_purity_data
from gex import process_gex_data
from entrezgene_id_to_mean_and_std_gex_mapping import process_entrezgene_id_to_mean_and_std_gex_mapping_data
from gene_predictability import process_gene_predictability_data

from utils import get_dfs_with_intersecting_sample_ids, get_dfs_with_intersecting_columns


data_dir = "/cluster/scratch/aarslan/cna2gex_data"
raw_folder_name = "raw"
processed_folder_name = "processed"
os.makedirs(os.path.join(data_dir, processed_folder_name), exist_ok=True)
tumor_sample_ids = ["0" + str(i) for i in range(1, 10)]


if __name__ == "__main__":
    thresholded_cna_df, unthresholded_cna_df = process_cna_data(data_dir=data_dir, raw_folder_name=raw_folder_name, processed_folder_name=processed_folder_name)
    cancer_type_df = process_cancer_type_data(data_dir=data_dir, raw_folder_name=raw_folder_name, processed_folder_name=processed_folder_name)
    tumor_purity_df = process_tumor_purity_data(data_dir=data_dir, raw_folder_name=raw_folder_name, tumor_sample_ids=tumor_sample_ids)
    gex_df = process_gex_data(data_dir=data_dir, processed_folder_name=processed_folder_name, tumor_sample_ids=tumor_sample_ids)
    gex_df, unthresholded_cna_df, thresholded_cna_df, tumor_purity_df, cancer_type_df = get_dfs_with_intersecting_sample_ids(dfs=[gex_df, unthresholded_cna_df, thresholded_cna_df, tumor_purity_df, cancer_type_df])
    gex_df, unthresholded_cna_df, thresholded_cna_df = get_dfs_with_intersecting_columns(dfs=[gex_df, unthresholded_cna_df, thresholded_cna_df])

    rppa_df = process_rppa_data(data_dir=data_dir, raw_folder_name=raw_folder_name, processed_folder_name=processed_folder_name, intersecting_columns=gex_df.columns.tolist(), intersecting_sample_ids=gex_df["sample_id"].tolist())
    rppa_df.to_csv(os.path.join(data_dir, processed_folder_name, "rppa.tsv"), sep="\t", index=False)
    print("rppa_df.shape:", rppa_df.shape)
    del rppa_df

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
    del cancer_type_df

    cancer_type_one_hot_df = pd.get_dummies(data=cancer_type_df, columns=["cancer_type"])
    cancer_type_one_hot_df.to_csv(os.path.join(data_dir, processed_folder_name, "cancer_type_one_hot.tsv"), sep="\t", index=False)
    print("cancer_type_one_hot_df.shape", cancer_type_one_hot_df.shape)
    del cancer_type_one_hot_df

    gex_df.to_csv(os.path.join(data_dir, processed_folder_name, "gex.tsv"), sep="\t", index=False)
    print("gex_df.shape:", gex_df.shape)

    entrezgene_id_to_mean_and_std_gex_mapping_df = process_entrezgene_id_to_mean_and_std_gex_mapping_data(gex_df=gex_df)
    entrezgene_id_to_mean_and_std_gex_mapping_df.to_csv(os.path.join(data_dir, processed_folder_name, "entrezgene_id_to_mean_and_std_gex_mapping.tsv"), sep="\t")
    print("entrezgene_id_to_mean_and_std_gex_mapping_df.shape:", entrezgene_id_to_mean_and_std_gex_mapping_df.shape)

    gene_predictability_df = process_gene_predictability_data(gex_df=gex_df, entrezgene_id_to_mean_and_std_gex_mapping_df=entrezgene_id_to_mean_and_std_gex_mapping_df)
    gene_predictability_df.to_csv(os.path.join(data_dir, processed_folder_name, "entrezgene_id_to_aug_adg_ddg_dug_ratios_mapping.tsv"), sep="\t")
    print("gene_predictability_df.shape:", gene_predictability_df.shape)
