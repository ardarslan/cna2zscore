cd src

for CANCER_TYPE in all  # all blca
    do
    for DATASET in avggexsubtype2gex  # avggexsubtype2gex cnapurity2gex rppa2gex
        do
        for HIDDEN_DIMENSION in 5000  # 1000 2500 5000
            do
            for NORMALIZE_OUTPUT in true  # false true
                do
                # bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --hidden_dimension $HIDDEN_DIMENSION --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers 1 --use_residual_connection false
                # bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --hidden_dimension $HIDDEN_DIMENSION --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers 2 --use_residual_connection false
                bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --hidden_dimension $HIDDEN_DIMENSION --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers 2 --use_residual_connection true
                done
            done
        done
    done
