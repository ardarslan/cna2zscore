cd src


for DATASET in 'unthresholdedcna2gex' 'unthresholdedcnapurity2gex' 'thresholdedcnapurity2gex' 'thresholdedcnapurity2gex' 'rppa2gex'; do
    for CANCER_TYPE in 'blca' 'all'; do
        for MODEL in 'linear' 'mlp' 'rescon_mlp'; do
            if [[ $MODEL = 'linear' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                declare -a HIDDEN_DIMENSION_OPTIONS=(0)
            elif [[ $MODEL = 'mlp' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                declare -a HIDDEN_DIMENSION_OPTIONS=(2500 5000 10000)
            elif [[ $MODEL = 'rescon_mlp' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false true)
                declare -a HIDDEN_DIMENSION_OPTIONS=(2500 5000 10000)
            else
                echo "MODEL is not a valid $MODEL."
                exit 1
            fi

            for RESCON_DIAGONAL_W in "${RESCON_DIAGONAL_W_OPTIONS}"; do
                for NUM_NONLINEAR_LAYERS in 1 2 3; do
                    for HIDDEN_DIMENSION in "${HIDDEN_DIMENSION_OPTIONS}"; do

                        # No regularization
                        sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_coeff 0.0 --l2_reg_coeff 0.0"

                        # L1 regularization
                        for L1_REG_COEFF in 0.001 0.01; do
                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_coeff $L1_REG_COEFF --l2_reg_coeff 0.0"
                        done

                        # L2 regularization
                        for L2_REG_COEFF in 0.001 0.01; do
                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=16384 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_coeff 0.0 --l2_reg_coeff $L2_REG_COEFF"
                        done
                    done
                done
            done
        done
    done
done
