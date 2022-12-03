cd src


for DATASET in 'unthresholdedcnapurity2gex' 'thresholdedcnapurity2gex' 'rppa2gex'; do
    for NORMALIZE_OUTPUT in false true; do
        for HIDDEN_DIMENSION in 2500 5000 10000; do
            for NUM_HIDDEN_LAYERS in 0 1 2; do
                for CANCER_TYPE in 'blca' 'all'; do

                    # No regularization
                    bsub -n 4 -W 24:00 -R "rusage[mem=16378, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff 0.0 --l2_reg_coeff 0.0

                    # Dropout
                    for DROPOUT in 0.25 0.5; do
                        bsub -n 4 -W 24:00 -R "rusage[mem=16378, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout $DROPOUT --l1_reg_coeff 0.0 --l2_reg_coeff 0.0
                    done

                    # L1 regularization
                    for L1_REG_COEFF in 0.001 0.01; do
                        bsub -n 4 -W 24:00 -R "rusage[mem=16378, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff $L1_REG_COEFF --l2_reg_coeff 0.0
                    done

                    # L2 regularization
                    for L2_REG_COEFF in 0.001 0.01; do
                        bsub -n 4 -W 24:00 -R "rusage[mem=16378, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff 0.0 --l2_reg_coeff $L2_REG_COEFF
                    done

                done
            done
        done
    done
done
