#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from rpy2.robjects.packages import importr, data
from rpy2.robjects import r
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.pandas2ri import rpy2py


# In[3]:


data_dir = "../data"
raw_folder_name = "raw"
processed_folder_name = "processed"
gex_file_name = "tcga_gene_expected_count"
output_file_name = "ensembl_id_to_entrezgene_id_mapping.tsv"


# In[4]:


cna_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes"), sep="\t")


# In[5]:


cna_df_hgnc_symbols = list(set(cna_df["Sample"].apply(lambda x: x.split("|")[0]).tolist()))
hgnc_to_entrezgene_id_mapping = dict(pd.read_csv(os.path.join(data_dir, raw_folder_name, "hgnc_to_entrezgene_id_mapping.tsv"), sep="\t").values)
cna_df_entrezgene_ids = set()
for cna_df_hgnc_symbol in cna_df_hgnc_symbols:
    if cna_df_hgnc_symbol in hgnc_to_entrezgene_id_mapping.keys():
        cna_df_entrezgene_ids.add(hgnc_to_entrezgene_id_mapping[cna_df_hgnc_symbol])
del cna_df


# In[6]:


gex_df = pd.read_csv(os.path.join(data_dir, raw_folder_name, "tcga_gene_expected_count"), sep="\t")
gex_df_ensembl_ids = tuple(set(gex_df["sample"].apply(lambda x: x.split(".")[0]).tolist()))
del gex_df


# In[7]:


r.library("biomaRt")
mart = r.useMart(biomart="ensembl", dataset="hsapiens_gene_ensembl")
ensembl_entrezgene_r_df = r.getBM(attributes = StrVector(("ensembl_gene_id", "entrezgene_id", )),
                                  filters = StrVector(("ensembl_gene_id", )),
                                  values = StrVector(gex_df_ensembl_ids),
                                  mart = mart)
ensembl_entrezgene_pandas_df = rpy2py(ensembl_entrezgene_r_df)
ensembl_entrezgene_pandas_df = ensembl_entrezgene_pandas_df[ensembl_entrezgene_pandas_df["entrezgene_id"] > 0]
ensembl_entrezgene_mapping = dict(ensembl_entrezgene_pandas_df.values)

gex_df_entrezgene_ids = set(ensembl_entrezgene_pandas_df["entrezgene_id"].tolist())


# In[8]:


missing_entrezgene_ids = cna_df_entrezgene_ids - gex_df_entrezgene_ids


# In[20]:


missing_entrezgene_ids = list(missing_entrezgene_ids)[list(missing_entrezgene_ids).index(106479253):]


# In[21]:


driver = webdriver.Chrome(ChromeDriverManager().install())

string_to_search = '"https://www.ensembl.org/Homo_sapiens/geneview?gene='

for missing_entrezgene_id in tqdm(missing_entrezgene_ids):
    try:
        driver.get(f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={missing_entrezgene_id}")
        content = driver.page_source
        content = content[content.index(string_to_search)+len(string_to_search):]
        content = content[:content.index('"')]
        ensembl_entrezgene_mapping[content] = missing_entrezgene_id
    except Exception as e:
        print(f"Could not fetch information of the gene with entrezgene ID {missing_entrezgene_id}.")

ensembl_entrezgene_pandas_df = pd.DataFrame.from_dict(
    {
       "ensembl_id": ensembl_entrezgene_mapping.keys(),
       "entrezgene_id": ensembl_entrezgene_mapping.values()
    }
)


# In[22]:


ensembl_entrezgene_pandas_df.to_csv(os.path.join(data_dir, processed_folder_name, output_file_name), index=False, sep="\t")


# In[23]:


driver.close()


# In[ ]:





# In[ ]:
