cd src

for CANCER_TYPE in blca all
    for DATASET in cnapurity2gex rppa2gex avggexsubtype2gex
        for HIDDEN_DIMENSION in 2000 5000 10000 max
            for NORMALIZE_INPUT_OUTPUT in false true
                bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" python main.py --cancer_type CANCER_TYPE --dataset DATASET --hidden_dimension HIDDEN_DIMENSION --normalize_input NORMALIZE_INPUT_OUTPUT --normalize_output NORMALIZE_INPUT_OUTPUT --num_hidden_layers 1 --use_residual_connection false
                bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" python main.py --cancer_type CANCER_TYPE --dataset DATASET --hidden_dimension HIDDEN_DIMENSION --normalize_input NORMALIZE_INPUT_OUTPUT --normalize_output NORMALIZE_INPUT_OUTPUT --num_hidden_layers 2 --use_residual_connection false
                bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" python main.py --cancer_type CANCER_TYPE --dataset DATASET --hidden_dimension HIDDEN_DIMENSION --normalize_input NORMALIZE_INPUT_OUTPUT --normalize_output NORMALIZE_INPUT_OUTPUT --num_hidden_layers 2 --use_residual_connection true
