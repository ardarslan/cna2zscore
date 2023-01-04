cd src

for MODEL in 'linear' 'linear_per_chromosome_24' 'mlp' 'mlp_per_chromosome_24' 'transformer'; do # 'linear_per_chromosome_all' 'mlp_per_chromosome_all' 'rescon_mlp'; do
    for CANCER_TYPE in 'all'; do
        for DATASET in 'unthresholdedcnapurity2gex' 'rppa2gex'; do

            if [[ $DATASET = 'rppa2gex' ]]; then
                declare -a GENE_TYPE_OPTIONS=("rppa_genes")
            else
                declare -a GENE_TYPE_OPTIONS=("5000_highly_expressed_genes" "rppa_genes")
            fi

            for GENE_TYPE in "${GENE_TYPE_OPTIONS[@]}"; do
                if [[ $MODEL = 'linear' || $MODEL = 'linear_per_chromosome_all' || $MODEL = 'linear_per_chromosome_24' ]]; then
                    declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                    declare -a HIDDEN_DIMENSION_OPTIONS=(0.0)
                    declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                    declare -a L1_REG_DIAGONAL_COEFF_OPTIONS=(0.001 0.01 0.1)
                    declare -a L2_REG_DIAGONAL_COEFF_OPTIONS=(0.001 0.01 0.1)
                    declare -a L1_REG_NONDIAGONAL_COEFF_OPTIONS=(0.01 0.1)
                    declare -a L2_REG_NONDIAGONAL_COEFF_OPTIONS=(0.01 0.1)
                elif [[ $MODEL = 'mlp' ]]; then
                    declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                    declare -a HIDDEN_DIMENSION_OPTIONS=(0.5 1.0)
                    declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                    declare -a L1_REG_COEFF_OPTIONS=(0.01 0.1)
                    declare -a L2_REG_COEFF_OPTIONS=(0.01 0.1)
                elif [[ $MODEL = 'mlp_per_chromosome_all' || $MODEL = 'mlp_per_chromosome_24' ]]; then
                    declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                    declare -a HIDDEN_DIMENSION_OPTIONS=(0.5 1.0)
                    declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                    declare -a L1_REG_COEFF_OPTIONS=(0.01 0.1)
                    declare -a L2_REG_COEFF_OPTIONS=(0.01 0.1)
                elif [[ $MODEL = 'rescon_mlp' ]]; then
                    declare -a RESCON_DIAGONAL_W_OPTIONS=(false true)
                    declare -a HIDDEN_DIMENSION_OPTIONS=(0.5 1.0)
                    declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                    declare -a L1_REG_COEFF_OPTIONS=(0.01 0.1)
                    declare -a L2_REG_COEFF_OPTIONS=(0.01 0.1)
                elif [[ $MODEL = 'transformer' ]]; then
                    declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                    declare -a HIDDEN_DIMENSION_OPTIONS=(0.0)
                    declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                    declare -a L1_REG_COEFF_OPTIONS=(0.01 0.1)
                    declare -a L2_REG_COEFF_OPTIONS=(0.01 0.1)
                else
                    echo "MODEL is not a valid $MODEL."
                    exit 1
                fi


                for RESCON_DIAGONAL_W in "${RESCON_DIAGONAL_W_OPTIONS[@]}"; do
                    for NUM_NONLINEAR_LAYERS in "${NUM_NONLINEAR_LAYERS_OPTIONS[@]}"; do
                        for HIDDEN_DIMENSION in "${HIDDEN_DIMENSION_OPTIONS[@]}"; do
                            for LEARNING_RATE in 0.0001 0.001; do

                                # No regularization
                                sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout 0.0"

                                # L1 regularization
                                if [[ $MODEL = "linear" || $MODEL = "linear_per_chromosome_all" || $MODEL = "linear_per_chromosome_24" ]]; then
                                    for L1_REG_DIAGONAL_COEFF in "${L1_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                        for L1_REG_NONDIAGONAL_COEFF in "${L1_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                            if [[ $(echo "$L1_REG_NONDIAGONAL_COEFF < $L1_REG_DIAGONAL_COEFF" |bc -l) ]]; then
                                                continue
                                            else
                                                sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff $L1_REG_DIAGONAL_COEFF --l1_reg_nondiagonal_coeff $L1_REG_NONDIAGONAL_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout 0.0"
                                            fi
                                        done
                                    done
                                else
                                    for L1_REG_COEFF in "${L1_REG_COEFF_OPTIONS[@]}"; do
                                        sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff $L1_REG_COEFF --l1_reg_nondiagonal_coeff $L1_REG_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout 0.0"
                                    done
                                fi

                                # L2 regularization
                                if [[ $MODEL = "linear" || $MODEL = "linear_per_chromosome_all" || $MODEL = "linear_per_chromosome_24" ]]; then
                                    for L2_REG_DIAGONAL_COEFF in "${L2_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                        for L2_REG_NONDIAGONAL_COEFF in "${L2_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                            if [[ $(echo "$L2_REG_NONDIAGONAL_COEFF < $L2_REG_DIAGONAL_COEFF" |bc -l) ]]; then
                                                continue
                                            else
                                                sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_DIAGONAL_COEFF --l2_reg_nondiagonal_coeff $L2_REG_NONDIAGONAL_COEFF --dropout 0.0"
                                            fi
                                        done
                                    done
                                else
                                    for L2_REG_COEFF in "${L2_REG_COEFF_OPTIONS[@]}"; do
                                        sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_COEFF --l2_reg_nondiagonal_coeff $L2_REG_COEFF --dropout 0.0"
                                    done
                                fi

                                # Dropout
                                for DROPOUT in 0.25; do
                                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT"
                                done

                            done
                        done
                    done
                done
            done
        done
    done
done
