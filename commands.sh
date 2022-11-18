cd src

for CANCER_TYPE in blca all
    do
    for DATASET in cnapurity2gex rppa2gex avggexsubtype2gex
        do
        for HIDDEN_DIMENSION in 2000 5000 10000 max
            do
            for NORMALIZE_INPUT_OUTPUT in false true
                do
                bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" python main.py --cancer_type CANCER_TYPE --dataset DATASET --hidden_dimension HIDDEN_DIMENSION --normalize_input NORMALIZE_INPUT_OUTPUT --normalize_output NORMALIZE_INPUT_OUTPUT --num_hidden_layers 1 --use_residual_connection false
                bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" python main.py --cancer_type CANCER_TYPE --dataset DATASET --hidden_dimension HIDDEN_DIMENSION --normalize_input NORMALIZE_INPUT_OUTPUT --normalize_output NORMALIZE_INPUT_OUTPUT --num_hidden_layers 2 --use_residual_connection false
                bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" python main.py --cancer_type CANCER_TYPE --dataset DATASET --hidden_dimension HIDDEN_DIMENSION --normalize_input NORMALIZE_INPUT_OUTPUT --normalize_output NORMALIZE_INPUT_OUTPUT --num_hidden_layers 2 --use_residual_connection true
                done
            done
        done
    done
