cd src


for DATASET in 'unthresholdedcna2gex' 'unthresholdedcnapurity2gex' 'thresholdedcnapurity2gex' 'thresholdedcnapurity2gex' 'rppa2gex'; do
    for CANCER_TYPE in 'blca' 'all'; do
        for MODEL in 'linear_model' 'mlp'; do
            if [[ $MODEL -eq 'linear_model' ]]; then
                declare -a NUM_HIDDEN_LAYERS_OPTIONS=(0)
                declare -a HIDDEN_DIMENSION=(0)
            elif [[ $MODEL -eq 'mlp' ]]; then
                declare -a NUM_HIDDEN_LAYERS_OPTIONS=(0 1 2)
                declare -a HIDDEN_DIMENSION=(2500 5000 10000 'max')
            else
                echo "$MODEL is not a valid MODEL."
                exit 1
            fi

            for NUM_HIDDEN_LAYERS in "${NUM_HIDDEN_LAYERS_OPTIONS[@]}"; do
                for HIDDEN_DIMENSION in "${HIDDEN_DIMENSION_OPTIONS}"; do

                    # No regularization
                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff 0.0 --l2_reg_coeff 0.0"

                    # Dropout
                    for DROPOUT in 0.3; do
                        sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout $DROPOUT --l1_reg_coeff 0.0 --l2_reg_coeff 0.0"
                    done

                    # L1 regularization
                    for L1_REG_COEFF in 0.001 0.01; do
                        sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff $L1_REG_COEFF --l2_reg_coeff 0.0"
                    done

                    # L2 regularization
                    for L2_REG_COEFF in 0.001 0.01; do
                        sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff 0.0 --l2_reg_coeff $L2_REG_COEFF"
                    done

                done
            done
        done
    done
done
