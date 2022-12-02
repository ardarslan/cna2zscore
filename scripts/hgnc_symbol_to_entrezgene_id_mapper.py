#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.pandas2ri import rpy2py


# In[3]:


# import sys
# !{sys.executable} -m pip install webdriver-manager

# utils = importr('utils')
# utils.install_packages('BiocManager', repos="https://cloud.r-project.org")
# r('BiocManager::install("biomaRt")')
r('library(biomaRt)')


# In[5]:


data_dir = "../data"
raw_folder_name = "raw"
cna_file_name = "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"
output_file_name = "hgnc_to_entrezgene_id_mapping.tsv"


# In[ ]:


cna_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, cna_file_name), sep="\t")
cna_df = cna_df.rename(columns={"Gene Symbol": "hgnc_symbol"})

for index, row in cna_df.iterrows():
    gene_symbol_splitted = row["hgnc_symbol"].split("|")
    if len(gene_symbol_splitted) == 1:
        cna_df.at[index, "ensembl_id"] = np.NaN
        cna_df.at[index, "hgnc_symbol"] = gene_symbol_splitted[0]
    elif len(gene_symbol_splitted) == 2:
        cna_df.at[index, "hgnc_symbol"] = gene_symbol_splitted[0]
        cna_df.at[index, "ensembl_id"] = gene_symbol_splitted[1]
    else:
        raise Exception("gene_symbol_splitted has more than 1 '|'s")

hgnc_symbols = tuple(set(cna_df["hgnc_symbol"]))

mart = r.useMart(biomart="ensembl", dataset="hsapiens_gene_ensembl")
hgnc_entrezgene_r_df = r.getBM(attributes = StrVector(("hgnc_symbol", "entrezgene_id", )),
                               filters = StrVector(("hgnc_symbol", )),
                               values = StrVector(hgnc_symbols),
                               mart = mart)

hgnc_entrezgene_pandas_df = rpy2py(hgnc_entrezgene_r_df)

hgnc_entrezgene_pandas_df = hgnc_entrezgene_pandas_df[(~pd.isnull(hgnc_entrezgene_pandas_df["hgnc_symbol"])) & (hgnc_entrezgene_pandas_df["entrezgene_id"] >= 0)]

hgnc_entrezgene_mapping = dict(hgnc_entrezgene_pandas_df.values)

hgnc_entrezgene_mapping_keys = hgnc_entrezgene_mapping.keys()


# In[ ]:


g


# In[ ]:


driver = webdriver.Chrome(ChromeDriverManager().install())

string_to_search = '"https://www.ncbi.nlm.nih.gov/gene/'

for hgnc_symbol in tqdm(set(cna_df["hgnc_symbol"]) - set(hgnc_entrezgene_mapping_keys)):
    try:
        driver.get(f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={hgnc_symbol}")
        content = driver.page_source
        content = content[content.index(string_to_search)+len(string_to_search):]
        content = content[:content.index('"')]
        hgnc_entrezgene_mapping[hgnc_symbol] = int(content)
    except Exception as e:
        print(f"Could not fetch information of the gene with hgnc symbol {hgnc_symbol}.")

hgnc_entrezgene_pandas_df = pd.DataFrame.from_dict(
    {
       "hgnc_symbol": hgnc_entrezgene_mapping.keys(),
       "entrezgene_id": hgnc_entrezgene_mapping.values()
    }
)


# In[ ]:


hgnc_entrezgene_pandas_df.to_csv(os.path.join(data_dir, raw_folder_name, output_file_name), index=False, sep="\t")


# In[ ]:


driver.close()

