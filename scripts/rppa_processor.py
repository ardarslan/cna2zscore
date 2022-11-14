import os
import pandas as pd

data_dir = "../data"
raw_folder_name = "raw"
processed_folder_name = "processed"
rppa_file_name = "TCGA-RPPA-pancan-clean.txt"
output_file_name = "rppa_df.tsv"

rppa_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, rppa_file_name), sep="\t")
rppa_df = rppa_df.drop(columns=["TumorType"])
rppa_df = rppa_df.rename(columns={"SampleID": "sample_id"})

tumor_sample_ids = ["0" + str(i) for i in range(1, 10)]

# We take only the tumor samples.
# If there are multiple samples which have the same project id, TSS id, participant id, and sample id,
# then we take the one with lexicographically first vial id. If there is also a tie in vial ids, then we
# take the one with lexicographically first (portion id, analyte id).

sample_id_dict = {}
for sample_id in rppa_df["sample_id"]:
    if sample_id.split("-")[3][:2] not in tumor_sample_ids:
        continue
    sample_id_first_15 = sample_id[:15]
    if sample_id_first_15 in sample_id_dict.keys():
        if sample_id < sample_id_dict[sample_id_first_15]:
            sample_id_dict[sample_id_first_15] = sample_id
        else:
            continue
    else:
        sample_id_dict[sample_id_first_15] = sample_id

samples_before = rppa_df.shape[0]
rppa_df = rppa_df[rppa_df["sample_id"].isin(list(sample_id_dict.values()))]
samples_after = rppa_df.shape[1]
print(f"Number of samples dropped {samples_before - samples_after}.")

rppa_df["sample_id"] = rppa_df["sample_id"].apply(lambda x: x[:15])

proteins_before = rppa_df.shape[1]
rppa_df = rppa_df.dropna(axis=1)
proteins_after = rppa_df.shape[1]
print(f"Number of proteins dropped {proteins_before - proteins_after}.")

rppa_df.index = rppa_df["sample_id"].values
rppa_df = rppa_df.drop(columns=["sample_id"])

rppa_df.to_csv(os.path.join(data_dir, processed_folder_name, output_file_name), sep="\t")
