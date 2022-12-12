import pandas as pd


def process_entrezgene_id_to_mean_and_std_gex_mapping_data(gex_df: pd.DataFrame, thresholded_cna_df: pd.DataFrame) -> pd.DataFrame:
    mean_gex = gex_df.drop(columns=["sample_id"]).values.mean(axis=0, where=(thresholded_cna_df.drop(columns=["sample_id"]).values == 0)).ravel()
    std_gex = gex_df.drop(columns=["sample_id"]).values.std(axis=0, where=(thresholded_cna_df.drop(columns=["sample_id"]).values == 0)).ravel()
    entrezgene_id_to_mean_and_std_gex_mapping_df = pd.DataFrame.from_dict({"entrezgene_id": gex_df.drop(columns=["sample_id"]).columns.tolist(), "mean_gex": mean_gex.tolist(), "std_gex": std_gex.tolist()})
    return entrezgene_id_to_mean_and_std_gex_mapping_df
