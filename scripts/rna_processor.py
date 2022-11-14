import os
import pandas as pd

data_dir = "../data"
raw_folder_name = "raw"
processed_folder_name = "processed"

rna_file_name = "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
cna_blca_file_name = "TCGA.BLCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"
cna_lusc_file_name = "TCGA.LUSC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"
cna_ov_file_name = "TCGA.OV.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"

output_blca_file_name = "rna_blca.tsv"
output_blca_lusc_ov_file_name = "rna_blca_lusc_ov.tsv"

rna_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, rna_file_name), sep="\t")
tumor_sample_ids = ["0" + str(i) for i in range(1, 10)]

# We take only the tumor samples.
# If there are multiple samples which have the same project id, TSS id, participant id, and sample id,
# then we take the one with lexicographically first vial id. If there is also a tie in vial ids, then we
# take the one with (numerically, lexicographically) first (portion id, analyte id).

column_dict = {}
for column in rna_df.columns:
    if column == "gene_id":
        continue

    if column.split("-")[3][:2] not in tumor_sample_ids:
        continue

    column_first_15 = column[:15]
    if column_first_15 in column_dict.keys():
        if column < column_dict[column_first_15]:
            column_dict[column_first_15] = column
        else:
            continue
    else:
        column_dict[column_first_15] = column

columns_before = rna_df.shape[0]
rna_df = rna_df[["gene_id"] + list(column_dict.values())]
columns_after = rna_df.shape[1]
print(f"Number of samples dropped: {columns_before - columns_after}")

rna_df.columns = rna_df.columns.map(lambda x: x[:15])

# Drop the genes with nan values
rows_before = rna_df.shape[0]
rna_df = rna_df.dropna()
rows_after = rna_df.shape[0]

print(f"Number of genes dropped: {rows_before - rows_after}")

rna_df["gene_id"] = rna_df["gene_id"].apply(lambda x: x.split("|")[1])
rna_df = rna_df.T
rna_df.columns = rna_df.loc["gene_id", :].values
rna_df = rna_df.drop("gene_id")

rna_df.to_csv(os.path.join(data_dir, processed_folder_name, output_file_name), sep="\t")
