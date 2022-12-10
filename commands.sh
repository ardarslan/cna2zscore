cd src


for DATASET in 'unthresholdedcna2gex' 'unthresholdedcnapurity2gex' 'thresholdedcnapurity2gex' 'thresholdedcnapurity2gex' 'rppa2gex'; do
    for CANCER_TYPE in 'blca' 'all'; do
        for NUM_NONLINEAR_LAYERS in (0 1 2 3); do
            if [[ $NUM_NONLINEAR_LAYERS -eq 0 ]]; then
                declare -a HIDDEN_DIMENSION_OPTIONS=(0)
            else
                declare -a HIDDEN_DIMENSION_OPTIONS=(2500 5000 10000 'max')
            fi

            for HIDDEN_DIMENSION in "${HIDDEN_DIMENSION_OPTIONS}"; do

                # No regularization
                sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff 0.0 --l2_reg_coeff 0.0"

                # Dropout
                for DROPOUT in 0.3; do
                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout $DROPOUT --l1_reg_coeff 0.0 --l2_reg_coeff 0.0"
                done

                # L1 regularization
                for L1_REG_COEFF in 0.001 0.01; do
                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff $L1_REG_COEFF --l2_reg_coeff 0.0"
                done

                # L2 regularization
                for L2_REG_COEFF in 0.001 0.01; do
                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --dropout 0.0 --l1_reg_coeff 0.0 --l2_reg_coeff $L2_REG_COEFF"
                done

            done
        done
    done
done
