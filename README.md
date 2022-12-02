# Install Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Close the current terminal and open a new one.

# Load modules

```
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
```

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

Run the following lines to process raw data:

```
cd scripts
python3 ensembl_id_to_entrezgene_id_mapper.py
python3 hgnc_symbol_to_entrezgene_id_mapper.py
bsub -n 8 -W 04:00 -R "rusage[mem=32768, ngpus_excl_p=1]" python data_processor.py
cd ..
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

DATASET: thresholdedcnapurity2gex | unthresholdedcnapurity2gex | thresholdedcna2gex | unthresholdedcna2gex | rppa2gex

CANCER_TYPE: blca | all

NORMALIZE_INPUT: true | false

NORMALIZE_OUTPUT: true | false

HIDDEN_DIMENSION: {int} | max | min | mean

NUM_HIDDEN_LAYERS: {int}

USE_RESIDUAL_CONNECTION: true | false

```
cd src
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" python main.py --dataset DATASET --cancer_type CANCER_TYPE --normalize_input NORMALIZE_INPUT --normalize_output NORMALIZE_OUTPUT --num_hidden_layers NUM_HIDDEN_LAYERS --hidden_dimension HIDDEN_DIMENSION --use_residual_connection USE_RESIDUAL_CONNECTION
```
