# Install Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Close the current terminal and open a new one.

# Setup and activate Conda environment

```
conda env create -f environment.yml
conda activate cna2gex
```

# Prepare the data

Run the following notebooks in the given order:

```
nbs/hgnc_symbol_to_entrezgene_id_mapper.ipynb
nbs/data_processor.ipynb
```

# Run the code

```
python3 main.py
```
