import pandas as pd


def process_gene_predictability_data(gex_df: pd.DataFrame, entrezgene_id_to_mean_and_std_gex_mapping_df: pd.DataFrame) -> pd.DataFrame:
    gex_df = gex_df.drop(columns=["sample_id"])
    gex_df = pd.DataFrame(data=(gex_df.values - entrezgene_id_to_mean_and_std_gex_mapping_df["mean_gex"].values) / (entrezgene_id_to_mean_and_std_gex_mapping_df["std_gex"].values + 1e-10), columns=gex_df.columns.tolist())
    thresholded_cna_df = thresholded_cna_df.drop(columns=["sample_id"])

    aug_adg_ddg_dug_ratios = []

    for entrezgene_id in gex_df.columns:
        current_genes_z_scores = gex_df[entrezgene_id].values
        current_genes_thresholded_cna_values = thresholded_cna_df[entrezgene_id].values
        aug_count = 0
        ddg_count = 0
        adg_count = 0
        dug_count = 0

        for current_genes_current_z_score, current_genes_current_thresholded_cna_value in zip(current_genes_z_scores, current_genes_thresholded_cna_values):
            if (current_genes_current_thresholded_cna_value > 0) and (current_genes_current_z_score > 2):
                aug_count += 1

            if (current_genes_current_thresholded_cna_value > 0) and (current_genes_current_z_score < -2):
                adg_count += 1

            if (current_genes_current_thresholded_cna_value < 0) and (current_genes_current_z_score < -2):
                ddg_count += 1

            if (current_genes_current_thresholded_cna_value < 0) and (current_genes_current_z_score > 2):
                dug_count += 1

        aug_adg_ddg_dug_ratios.append((entrezgene_id,
                                       float(aug_count) / float(gex_df.shape[0]),
                                       float(adg_count) / float(gex_df.shape[0]),
                                       float(ddg_count) / float(gex_df.shape[0]),
                                       float(dug_count) / float(gex_df.shape[0])))

    aug_adg_ddg_dug_ratios_df = pd.DataFrame(data=aug_adg_ddg_dug_ratios, columns=["entrezgene_id", "aug_ratio", "adg_ratio", "ddg_ratio", "dug_ratio"])
    return aug_adg_ddg_dug_ratios_df
