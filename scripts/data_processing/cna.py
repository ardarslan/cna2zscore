import os
import numpy as np
import pandas as pd
import swifter


def process_cna_data(data_dir: str, raw_folder_name: str, processed_folder_name: str) -> pd.DataFrame:
    print("Processing CNA data...")

    thresholded_cna_file_name = "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"
    unthresholded_cna_file_name = "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"

    hgnc_symbol_to_entrezgene_id_mapping_file_name = "hgnc_to_entrezgene_id_mapping.tsv"
    hgnc_symbol_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, processed_folder_name, hgnc_symbol_to_entrezgene_id_mapping_file_name), sep="\t").values)

    gex_file_name = "tcga_RSEM_gene_tpm"
    gex_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, gex_file_name), sep="\t", usecols=["sample"])
    gex_ensembl_ids = frozenset(gex_df["sample"].tolist())
    del gex_df

    def process_cna_data_helper(cna_file_name):
        cna_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, cna_file_name), sep="\t")

        for index, row in cna_df.iterrows():
            sample_splitted = row["Sample"].split("|")
            if len(sample_splitted) == 1:
                cna_df.at[index, "ensembl_id"] = ""
                cna_df.at[index, "entrezgene_id"] = hgnc_symbol_to_entrezgene_id_mapping.get(sample_splitted[0], np.NaN)
            elif len(sample_splitted) == 2:
                cna_df.at[index, "ensembl_id"] = sample_splitted[1]
                cna_df.at[index, "entrezgene_id"] = hgnc_symbol_to_entrezgene_id_mapping.get(sample_splitted[0], np.NaN)
            else:
                raise Exception("sample_splitted has more than 1 '|'s")

        cna_df = cna_df[~pd.isnull(cna_df["entrezgene_id"])]
        cna_df["entrezgene_id"] = cna_df["entrezgene_id"].swifter.apply(lambda x: int(x))

        cna_df["ensembl_id_is_not_in_gex_ensembl_ids"] = cna_df["ensembl_id"].swifter.apply(lambda x: 1 * (x not in gex_ensembl_ids))

        def get_ensembl_version(ensembl_id):
            if pd.isnull(ensembl_id) or ensembl_id == "":
                return -1
            else:
                return int(ensembl_id.split(".")[-1])

        cna_df["ensembl_version"] = cna_df["ensembl_id"].swifter.apply(lambda ensembl_id: get_ensembl_version(ensembl_id))

        def select_one_row_per_entrezgene_id(x):
            return x.sort_values(by=["ensembl_id_is_not_in_gex_ensembl_ids", "ensembl_version"], ascending=True).iloc[0, :]

        cna_df = cna_df.swifter.groupby("entrezgene_id").apply(lambda x: select_one_row_per_entrezgene_id(x)).reset_index(drop=True)

        cna_df.drop(columns=["Sample", "ensembl_id", "ensembl_id_is_not_in_gex_ensembl_ids", "ensembl_version"], inplace=True)

        cna_df.set_index("entrezgene_id", inplace=True)

        cna_df = cna_df.T

        cna_df.reset_index(drop=False, inplace=True)
        cna_df = cna_df.rename_axis(None, axis=1)
        cna_df.rename(columns={"index": "sample_id"}, inplace=True)

        return cna_df

    thresholded_cna_df = process_cna_data_helper(cna_file_name=thresholded_cna_file_name)
    unthresholded_cna_df = process_cna_data_helper(cna_file_name=unthresholded_cna_file_name)

    print("Processed CNA data.")

    return thresholded_cna_df, unthresholded_cna_df
