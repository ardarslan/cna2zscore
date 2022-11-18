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
unzip raw.zip
rm -rf raw.zip
cd ..
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
unzip processed.zip
rm -rf processed.zip
cd ..
```

# Run the code

DATASET: cnapurity2gex | rppa2gex | avggexsubtype2gex

CANCER_TYPE: blca | lusc | ov | all

NORMALIZE_INPUT: true | false

NORMALIZE_OUTPUT: true | false

HIDDEN_DIMENSION: {int} | max | min | mean

NUM_HIDDEN_LAYERS: {int}

USE_RESIDUAL_CONNECTION: true | false

```
cd src
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python main.py --dataset DATASET --cancer_type CANCER_TYPE --normalize_input NORMALIZE_INPUT --normalize_output NORMALIZE_OUTPUT --num_hidden_layers NUM_HIDDEN_LAYERS --hidden_dimension HIDDEN_DIMENSION --use_residual_connection USE_RESIDUAL_CONNECTION
```
