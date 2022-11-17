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

# Download raw data and process it

Download raw data
```
mkdir data
cd data
gdown 1aJo5IdlI545lKniNjX33GD_cH4hurQ2Y
unzip raw.zip -d raw
rm -rf raw.zip
```

Run the following notebooks in the given order:

```
nbs/hgnc_symbol_to_entrezgene_id_mapper.ipynb
nbs/data_processor.ipynb
```

# Or download the processed data

```
mkdir data
cd data
gdown 1CPdjSSt7QhJZgpbPWmf_uFnWJgthP4l-
unzip processed.zip -d processed
rm -rf processed.zip
```

# Run the code

```
python3 main.py
```
