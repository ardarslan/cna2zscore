#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import pandas as pd
from tqdm import tqdm
import requests

from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.pandas2ri import rpy2py

# In[ ]:

# import sys
# !{sys.executable} -m pip install webdriver-manager

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# utils = importr('utils')
# utils.install_packages('BiocManager', repos="https://cloud.r-project.org")
# r('BiocManager::install("biomaRt")')
r('library(biomaRt)')

driver = webdriver.Chrome(ChromeDriverManager().install())

# In[ ]:

data_dir = "../../data"
raw_folder_name = "raw"
processed_folder_name = "processed"
cna_file_name = "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"
rppa_file_name = "TCGA-RPPA-pancan-clean.xena"
gex_file_name = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
breast_cancer_ipac_genes_file_name = "breast_cancer_ipac_genes.tsv"
breast_cancer_scc_genes_file_name = "breast_cancer_scc_genes.tsv"
protein_to_hgnc_mapping_file_name = "tcpa_to_ncbi_mapping.csv"
output_file_name = "hgnc_to_entrezgene_id_mapping.tsv"

# In[ ]:

# cna_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, cna_file_name), sep="\t")
# cna_df["Sample"] = cna_df["Sample"].apply(lambda x: x.split("|")[0])
# cna_df_hgnc_symbols = set(cna_df["Sample"].tolist())
# del cna_df

# # In[ ]:

# rppa_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, rppa_file_name), sep="\t")
# rppa_df.index = rppa_df["SampleID"].tolist()
# rppa_df.drop(columns=["SampleID"], inplace=True)
# rppa_df = rppa_df.T
# rppa_df.reset_index(drop=False, inplace=True)
# rppa_df.rename(columns={"index": "sample_id"}, inplace=True)
# rppa_df = rppa_df.dropna(axis=1)
# protein_to_hgnc_mapping_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, protein_to_hgnc_mapping_file_name), sep=",")
# protein_to_hgnc_mapping = dict(protein_to_hgnc_mapping_df[["TCPA Symbol", "NCBI Symbol 1"]].values)
# rppa_df_hgnc_symbols = set([protein_to_hgnc_mapping[column] for column in rppa_df.columns if column != "sample_id"])
# del rppa_df
# del protein_to_hgnc_mapping_df

# gex_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, gex_file_name), sep="\t")
# gex_df_hgnc_symbols = set([gene_id for gene_id in gex_df["sample"].tolist() if isinstance(gene_id, str) and (not gene_id.isdigit())])
# del gex_df

breast_cancer_ipac_genes_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, breast_cancer_ipac_genes_file_name), sep="\t")
breast_cancer_ipac_genes_hgnc_symbols = set(breast_cancer_ipac_genes_df["hgnc_symbol"].tolist())

breast_cancer_scc_genes_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, breast_cancer_scc_genes_file_name), sep="\t")
breast_cancer_scc_genes_hgnc_symbols = set(breast_cancer_scc_genes_df["hgnc_symbol"].tolist())

hgnc_symbols = cna_df_hgnc_symbols.union(gex_df_hgnc_symbols).union(rppa_df_hgnc_symbols).union(breast_cancer_ipac_genes_hgnc_symbols).union(breast_cancer_scc_genes_hgnc_symbols)

mart = r.useMart(biomart="ensembl", dataset="hsapiens_gene_ensembl")
r_df = r.getBM(attributes = StrVector(("hgnc_symbol", "entrezgene_id", "chromosome_name")),
               filters = StrVector(("hgnc_symbol", )),
               values = StrVector(tuple(hgnc_symbols)),
               mart = mart)

pandas_df = rpy2py(r_df)
pandas_df = pandas_df[~pd.isnull(pandas_df["hgnc_symbol"]) & \
                      ~pd.isnull(pandas_df["entrezgene_id"]) & \
                      ~pd.isnull(pandas_df["chromosome_name"]) & \
                      pandas_df["entrezgene_id"].apply(lambda x: x > 0)]

hgnc_entrezgene_id_mapping = dict(pandas_df[["hgnc_symbol", "entrezgene_id"]].values)
entrezgene_id_chromosome_name_mapping = dict(pandas_df[["entrezgene_id", "chromosome_name"]].values)

hgnc_symbols = list(hgnc_symbols - hgnc_entrezgene_id_mapping.keys())

for hgnc_symbol in tqdm(hgnc_symbols):
    try:
        string_to_search = '"https://www.ncbi.nlm.nih.gov/gene/'

        driver.get(f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={hgnc_symbol}")
        content = driver.page_source
        content = content[content.index(string_to_search)+len(string_to_search):]
        entrezgene_id = int(content[:content.index('"')])
        hgnc_entrezgene_id_mapping[hgnc_symbol] = entrezgene_id

        url = f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={hgnc_symbol}"
        page = requests.get(url).content.decode()
        string_to_search = '="constant dynamic-link">Chromosome '
        page = page[page.index(string_to_search)+len(string_to_search):]
        chromosome_name = page[:page.index(":")]
        entrezgene_id_chromosome_name_mapping[entrezgene_id] = chromosome_name
    except Exception as e:
        print(f"Error with hgnc_symbol: {hgnc_symbol}: {e}")

hgnc_entrezgene_id_mapping_df = pd.DataFrame.from_dict({"hgnc_symbol": hgnc_entrezgene_id_mapping.keys(), "entrezgene_id": hgnc_entrezgene_id_mapping.values()})
hgnc_entrezgene_id_mapping_df.to_csv(os.path.join(data_dir, processed_folder_name, "hgnc_to_entrezgene_id_mapping.tsv"), sep="\t", index=False)

entrezgene_id_chromosome_name_mapping_df = pd.DataFrame.from_dict({"entrezgene_id": entrezgene_id_chromosome_name_mapping.keys(), "chromosome_name": entrezgene_id_chromosome_name_mapping.values()})
entrezgene_id_chromosome_name_mapping_df.to_csv(os.path.join(data_dir, processed_folder_name, "entrezgene_id_chromosome_name_mapping.tsv"), sep="\t", index=False)
