#!/usr/bin/env python
# coding: utf-8

# In[1]:


development = False


# In[3]:


import os
import numpy as np
import pandas as pd
import swifter


# In[4]:


data_dir = "../data"
raw_folder_name = "raw"
if development:
    processed_folder_name = "development"
else:
    processed_folder_name = "processed"

os.makedirs(os.path.join(data_dir, processed_folder_name), exist_ok=True)


# In[5]:


tumor_sample_ids = ["0" + str(i) for i in range(1, 10)]


# # # Process RPPA Data

# # In[6]:

# print("Processing RPPA data...")
# rppa_file_name = "TCGA-RPPA-pancan-clean.xena"

# rppa_df = pd.read_csv(os.path.join(data_dir, "raw", rppa_file_name), sep="\t")
# rppa_df.index = rppa_df["SampleID"].tolist()
# rppa_df.drop(columns=["SampleID"], inplace=True)
# rppa_df = rppa_df.T
# rppa_df.reset_index(drop=False, inplace=True)
# rppa_df.rename(columns={"index": "sample_id"}, inplace=True)
# rppa_df = rppa_df.dropna(axis=1)

# if development:
#     print(rppa_df)

# print("Processed RPPA data.")

# # # Process Thresholded and Unthresholded CNA Data

# # In[7]:

# print("Processing CNA data...")
# thresholded_cna_file_name = "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"
# unthresholded_cna_file_name = "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"

# hgnc_symbol_to_entrezgene_id_mapping_file_name = "hgnc_to_entrezgene_id_mapping.tsv"
# hgnc_symbol_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, raw_folder_name, hgnc_symbol_to_entrezgene_id_mapping_file_name), sep="\t").values)

# gex_file_name = "tcga_gene_expected_count"
# gex_df = pd.read_csv(os.path.join(data_dir, "raw", gex_file_name), sep="\t", usecols=["sample"])
# gex_ensembl_ids = frozenset(gex_df["sample"].tolist())
# del gex_df

# def process_cna_df(cna_file_name):
#     if development:
#         cna_df = pd.read_csv(os.path.join(data_dir, "raw", cna_file_name), sep="\t", nrows=1000)
#     else:
#         cna_df = pd.read_csv(os.path.join(data_dir, "raw", cna_file_name), sep="\t")

#     for index, row in cna_df.iterrows():
#         sample_splitted = row["Sample"].split("|")
#         if len(sample_splitted) == 1:
#             cna_df.at[index, "ensembl_id"] = ""
#             cna_df.at[index, "entrezgene_id"] = hgnc_symbol_to_entrezgene_id_mapping.get(sample_splitted[0], np.NaN)
#         elif len(sample_splitted) == 2:
#             cna_df.at[index, "ensembl_id"] = sample_splitted[1]
#             cna_df.at[index, "entrezgene_id"] = hgnc_symbol_to_entrezgene_id_mapping.get(sample_splitted[0], np.NaN)
#         else:
#             raise Exception("sample_splitted has more than 1 '|'s")

#     cna_df = cna_df[~pd.isnull(cna_df["entrezgene_id"])]
#     cna_df["entrezgene_id"] = cna_df["entrezgene_id"].swifter.apply(lambda x: int(x))

#     cna_df["ensembl_id_is_not_in_gex_ensembl_ids"] = cna_df["ensembl_id"].swifter.apply(lambda x: 1 * (x not in gex_ensembl_ids))

#     def get_ensembl_version(ensembl_id):
#         if pd.isnull(ensembl_id) or ensembl_id == "":
#             return -1
#         else:
#             return int(ensembl_id.split(".")[-1])

#     cna_df["ensembl_version"] = cna_df["ensembl_id"].swifter.apply(lambda ensembl_id: get_ensembl_version(ensembl_id))

#     def select_one_row_per_entrezgene_id(x):
#         return x.sort_values(by=["ensembl_id_is_not_in_gex_ensembl_ids", "ensembl_version"], ascending=True).iloc[0, :]

#     cna_df = cna_df.swifter.groupby("entrezgene_id").apply(lambda x: select_one_row_per_entrezgene_id(x)).reset_index(drop=True)

#     cna_df.drop(columns=["Sample", "ensembl_id", "ensembl_id_is_not_in_gex_ensembl_ids", "ensembl_version"], inplace=True)

#     cna_df.set_index("entrezgene_id", inplace=True)

#     cna_df = cna_df.T

#     cna_df.reset_index(drop=False, inplace=True)
#     cna_df = cna_df.rename_axis(None, axis=1)
#     cna_df.rename(columns={"index": "sample_id"}, inplace=True)

#     return cna_df

# thresholded_cna_df = process_cna_df(cna_file_name=thresholded_cna_file_name)
# unthresholded_cna_df = process_cna_df(cna_file_name=unthresholded_cna_file_name)

# if development:
#     print(thresholded_cna_df)
#     print(unthresholded_cna_df)

# print("Processed CNA data.")

# # # Process Cancer Type Data

# # In[8]:

# print("Processing Cancer Type data...")

# cancer_type_file_name = "TCGA_phenotype_denseDataOnlyDownload.tsv"
# cancer_type_full_name_to_abbrreviation_mapping_file_name = "cancer_type_full_name_to_abbrreviation_mapping.tsv"

# cancer_type_full_name_to_abbrreviation_mapping = dict(pd.read_csv(os.path.join(data_dir, raw_folder_name, cancer_type_full_name_to_abbrreviation_mapping_file_name), sep="\t").values)

# cancer_type_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, cancer_type_file_name), sep="\t")
# cancer_type_df = cancer_type_df.rename(columns={"sample": "sample_id", "_primary_disease": "cancer_type"})
# cancer_type_df = cancer_type_df[["sample_id", "cancer_type"]]
# cancer_type_df["cancer_type"] = cancer_type_df["cancer_type"].swifter.apply(lambda x: cancer_type_full_name_to_abbrreviation_mapping[x])

# cancer_type_one_hot_df = pd.get_dummies(data=cancer_type_df, columns=["cancer_type"])

# if development:
#     print(cancer_type_df)
#     print(cancer_type_one_hot_df)

# print("Processed Cancer Type data...")

# # # Process Tumor Purity Data

# # In[9]:

# print("Processing Tumor Purity data...")

# tumor_purity_cpe_file_name = "tumor_purity.csv"
# tumor_purity_estimate_file_name = "tumor_purity_ESTIMATE.csv"

# tumor_purity_cpe_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, tumor_purity_cpe_file_name))
# tumor_purity_estimate_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, tumor_purity_estimate_file_name))

# tumor_sample_id_purity_mapping = {}
# for _, row in tumor_purity_cpe_df.iterrows():
#     sample_id = row["Sample.ID"]
#     purity_cpe = row["CPE"]
#     purity_estimate = row["ESTIMATE"]
#     purity_absolute = row["ABSOLUTE"]

#     if not pd.isnull(purity_cpe):
#         tumor_sample_id_purity_mapping[sample_id] = float(purity_cpe.replace(",", "."))
#         continue

#     if not pd.isnull(purity_absolute):
#         tumor_sample_id_purity_mapping[sample_id] = float(purity_absolute.replace(",", "."))
#         continue

#     if not pd.isnull(purity_estimate):
#         tumor_sample_id_purity_mapping[sample_id] = float(purity_estimate.replace(",", "."))
#         continue

# for _, row in tumor_purity_estimate_df.iterrows():
#     sample_id = row["NAME"]
#     purity_estimate = row["TumorPurity"]
#     if not pd.isnull(purity_estimate):
#         tumor_sample_id_purity_mapping[sample_id] = purity_estimate

# tumor_purity_df = pd.DataFrame.from_dict({"sample_id": tumor_sample_id_purity_mapping.keys(),
#                                           "purity": tumor_sample_id_purity_mapping.values()})

# sample_id_dict = {}
# for sample_id in tumor_purity_df["sample_id"].values:
#     if sample_id.split("-")[3][:2] not in tumor_sample_ids:
#         continue
#     sample_id_first_15 = sample_id[:15]
#     if sample_id_first_15 in sample_id_dict.keys():
#         if sample_id < sample_id_dict[sample_id_first_15]:
#             sample_id_dict[sample_id_first_15] = sample_id
#         else:
#             continue
#     else:
#         sample_id_dict[sample_id_first_15] = sample_id

# tumor_purity_df = tumor_purity_df[tumor_purity_df["sample_id"].swifter.apply(lambda x: x in list(sample_id_dict.values()))]
# tumor_purity_df["sample_id"] = tumor_purity_df["sample_id"].swifter.apply(lambda x: x[:15])

# if development:
#     print(tumor_purity_df)

# print("Processed Tumor Purity data.")

# # # Process GEX Data

# # In[10]:
print("Processing GEX data...")

ensembl_id_to_entrezgene_id_mapping_file_name = "ensembl_id_to_entrezgene_id_mapping.tsv"
gex_file_name = "tcga_gene_expected_count"

tumor_sample_ids = ["0" + str(i) for i in range(1, 10)]
ensembl_id_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, raw_folder_name, ensembl_id_to_entrezgene_id_mapping_file_name), sep="\t").values)

if development:
    gex_df = pd.read_csv(os.path.join(data_dir, "raw", gex_file_name), sep="\t", nrows=1000)
else:
    gex_df = pd.read_csv(os.path.join(data_dir, "raw", gex_file_name), sep="\t")

gex_df.rename(columns={"sample": "ensembl_id"}, inplace=True)
gex_df = gex_df[["ensembl_id"] + [column for column in gex_df.columns if column.split("-")[-1] in tumor_sample_ids]]
gex_df["ensembl_id"] = gex_df["ensembl_id"].swifter.apply(lambda x: x.split(".")[0]).tolist()
gex_df = gex_df[gex_df["ensembl_id"].swifter.apply(lambda x: x in ensembl_id_to_entrezgene_id_mapping.keys())]
gex_df["entrezgene_id"] = gex_df["ensembl_id"].swifter.apply(lambda x: ensembl_id_to_entrezgene_id_mapping[x]).tolist()
gex_df.drop(columns=["ensembl_id"], inplace=True)
gex_df.index = gex_df["entrezgene_id"].tolist()
gex_df.drop(columns=["entrezgene_id"], inplace=True)
gex_df = gex_df.T
gex_df.reset_index(drop=False, inplace=True)
gex_df.rename(columns={"index": "sample_id"}, inplace=True)

print(len(gex_df.columns), len(set(gex_df.columns)))

if development:
    print(gex_df)


print("Processed GEX data.")
# # Find intersecting sample IDs and columns

# In[11]:

# gex_sample_ids = set(gex_df["sample_id"].tolist())
# unthresholded_cna_sample_ids = set(unthresholded_cna_df["sample_id"].tolist())
# thresholded_cna_sample_ids = set(thresholded_cna_df["sample_id"].tolist())
# tumor_purity_sample_ids = set(tumor_purity_df["sample_id"].tolist())
# cancer_type_sample_ids = set(cancer_type_df["sample_id"].tolist())
# intersecting_sample_ids = gex_sample_ids.intersection(unthresholded_cna_sample_ids).intersection(thresholded_cna_sample_ids).intersection(tumor_purity_sample_ids).intersection(cancer_type_sample_ids)

# gex_gene_ids = set(gex_df.drop(columns=["sample_id"]).columns)
# unthresholded_cna_gene_ids = set(unthresholded_cna_df.drop(columns=["sample_id"]).columns)
# thresholded_cna_gene_ids = set(thresholded_cna_df.drop(columns=["sample_id"]).columns)
# intersecting_columns = ["sample_id"] + sorted(list(gex_gene_ids.intersection(unthresholded_cna_gene_ids).intersection(thresholded_cna_gene_ids)))


# # # Save data

# # In[12]:
# print("Saving data...")

# rppa_df = rppa_df[rppa_df["sample_id"].swifter.apply(lambda x: x in intersecting_sample_ids)]
# rppa_df = rppa_df.sort_values(by="sample_id")
# rppa_df.to_csv(os.path.join(data_dir, processed_folder_name, "rppa.tsv"), sep="\t", index=False)
# print("rppa_df.shape:", rppa_df.shape)

# thresholded_cna_df = thresholded_cna_df[thresholded_cna_df["sample_id"].swifter.apply(lambda x: x in intersecting_sample_ids)][intersecting_columns]
# thresholded_cna_df = thresholded_cna_df.sort_values(by="sample_id")
# thresholded_cna_df.to_csv(os.path.join(data_dir, processed_folder_name, "thresholded_cna.tsv"), sep="\t", index=False)
# print("thresholded_cna_df.shape:", thresholded_cna_df.shape)

# unthresholded_cna_df = unthresholded_cna_df[unthresholded_cna_df["sample_id"].swifter.apply(lambda x: x in intersecting_sample_ids)][intersecting_columns]
# unthresholded_cna_df = unthresholded_cna_df.sort_values(by="sample_id")
# unthresholded_cna_df.to_csv(os.path.join(data_dir, processed_folder_name, "unthresholded_cna.tsv"), sep="\t", index=False)
# print("unthresholded_cna_df.shape:", unthresholded_cna_df.shape)

# tumor_purity_df = tumor_purity_df[tumor_purity_df["sample_id"].swifter.apply(lambda x: x in intersecting_sample_ids)]
# tumor_purity_df = tumor_purity_df.sort_values(by="sample_id")
# tumor_purity_df.to_csv(os.path.join(data_dir, processed_folder_name, "tumor_purity.tsv"), sep="\t", index=False)
# print("tumor_purity_df.shape:", tumor_purity_df.shape)

# cancer_type_df = cancer_type_df[cancer_type_df["sample_id"].swifter.apply(lambda x: x in intersecting_sample_ids)]
# cancer_type_df = cancer_type_df.sort_values(by="sample_id")
# cancer_type_df.to_csv(os.path.join(data_dir, processed_folder_name, "cancer_type.tsv"), sep="\t", index=False)
# print("cancer_type_df.shape:", cancer_type_df.shape)

# cancer_type_one_hot_df = cancer_type_one_hot_df[cancer_type_one_hot_df["sample_id"].swifter.apply(lambda x: x in intersecting_sample_ids)]
# cancer_type_one_hot_df = cancer_type_one_hot_df.sort_values(by="sample_id")
# cancer_type_one_hot_df.to_csv(os.path.join(data_dir, processed_folder_name, "cancer_type_one_hot.tsv"), sep="\t", index=False)
# print("cancer_type_one_hot_df.shape", cancer_type_one_hot_df.shape)

# gex_df = gex_df[gex_df["sample_id"].swifter.apply(lambda x: x in intersecting_sample_ids)][intersecting_columns]
# gex_df = gex_df.sort_values(by="sample_id")
# gex_df.to_csv(os.path.join(data_dir, processed_folder_name, "gex.tsv"), sep="\t", index=False)
# print("gex_df.shape:", gex_df.shape)

# print("Saved data.")
