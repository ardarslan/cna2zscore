cd src

for MODEL in 'linear' 'linear_per_chromosome_24' 'mlp_per_chromosome_24' 'mlp' 'transformer' 'linear_per_chromosome_all' 'mlp_per_chromosome_all' 'rescon_mlp'; do
    for DATASET in 'unthresholdedcnapurity2gex' 'rppa2gex'; do
        for CANCER_TYPE in 'blca' 'all'; do
            if [[ $MODEL = 'linear' || $MODEL = 'linear_per_chromosome_all' || $MODEL = 'linear_per_chromosome_24' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                declare -a HIDDEN_DIMENSION_OPTIONS=(0)
                declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                declare -a L1_REG_DIAGONAL_COEFF_OPTIONS=(0.0 0.001 0.01 0.1)
                declare -a L2_REG_DIAGONAL_COEFF_OPTIONS=(0.0 0.001 0.01 0.1)
                declare -a L1_REG_NONDIAGONAL_COEFF_OPTIONS=(0.001 0.01 0.1)
                declare -a L2_REG_NONDIAGONAL_COEFF_OPTIONS=(0.001 0.01 0.1)
            elif [[ $MODEL = 'mlp' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                declare -a HIDDEN_DIMENSION_OPTIONS=(2500 5000 10000)
                declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(1 2 3)
                declare -a L1_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
                declare -a L2_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
            elif [[ $MODEL = 'mlp_per_chromosome_all' || $MODEL = 'mlp_per_chromosome_24' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                declare -a HIDDEN_DIMENSION_OPTIONS=(max)
                declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(1 2 3)
                declare -a L1_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
                declare -a L2_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
            elif [[ $MODEL = 'rescon_mlp' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false true)
                declare -a HIDDEN_DIMENSION_OPTIONS=(2500 5000 10000)
                declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(1 2 3)
                declare -a L1_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
                declare -a L2_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
            elif [[ $MODEL = 'transformer' ]]; then
                declare -a RESCON_DIAGONAL_W_OPTIONS=(false)
                declare -a HIDDEN_DIMENSION_OPTIONS=(0)
                declare -a NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                declare -a L1_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
                declare -a L2_REG_COEFF_OPTIONS=(0.001 0.01 0.1)
            else
                echo "MODEL is not a valid $MODEL."
                exit 1
            fi

            if [[ $MODEL = 'transformer' ]]; then
                declare -a GENE_TYPE_OPTIONS=("1000_highly_expressed_genes" "rppa_genes")
            elif [[ $DATASET = 'rppa2gex' ]]; then
                declare -a GENE_TYPE_OPTIONS=("rppa_genes")
            else
                declare -a GENE_TYPE_OPTIONS=("rppa_genes" "1000_highly_expressed_genes" "all_genes")
            fi

            for GENE_TYPE in "${GENE_TYPE_OPTIONS[@]}"; do
                for RESCON_DIAGONAL_W in "${RESCON_DIAGONAL_W_OPTIONS[@]}"; do
                    for NUM_NONLINEAR_LAYERS in "${NUM_NONLINEAR_LAYERS_OPTIONS[@]}"; do
                        for HIDDEN_DIMENSION in "${HIDDEN_DIMENSION_OPTIONS[@]}"; do

                            # No regularization
                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=2 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout 0.0"

                            # L1 regularization
                            if [[ $MODEL = "linear" || $MODEL = "linear_per_chromosome_all" || $MODEL = "linear_per_chromosome_24" ]]; then
                                for L1_REG_DIAGONAL_COEFF in "${L1_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                    for L1_REG_NONDIAGONAL_COEFF in "${L1_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                        if [[ $(echo "$L1_REG_NONDIAGONAL_COEFF < $L1_REG_DIAGONAL_COEFF" |bc -l) ]]; then
                                            continue
                                        else
                                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=2 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_diagonal_coeff $L1_REG_DIAGONAL_COEFF --l1_reg_nondiagonal_coeff $L1_REG_NONDIAGONAL_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout 0.0"
                                        fi
                                    done
                                done
                            else
                                for L1_REG_COEFF in "${L1_REG_COEFF_OPTIONS[@]}"; do
                                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=2 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_diagonal_coeff $L1_REG_COEFF --l1_reg_nondiagonal_coeff $L1_REG_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout 0.0"
                                done
                            fi

                            # L2 regularization
                            if [[ $MODEL = "linear" || $MODEL = "linear_per_chromosome_all" || $MODEL = "linear_per_chromosome_24" ]]; then
                                for L2_REG_DIAGONAL_COEFF in "${L2_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                    for L2_REG_NONDIAGONAL_COEFF in "${L2_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                        if [[ $(echo "$L2_REG_NONDIAGONAL_COEFF < $L2_REG_DIAGONAL_COEFF" |bc -l) ]]; then
                                            continue
                                        else
                                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=2 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_DIAGONAL_COEFF --l2_reg_nondiagonal_coeff $L2_REG_NONDIAGONAL_COEFF --dropout 0.0"
                                        fi
                                    done
                                done
                            else
                                for L2_REG_COEFF in "${L2_REG_COEFF_OPTIONS[@]}"; do
                                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=2 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_COEFF --l2_reg_nondiagonal_coeff $L2_REG_COEFF --dropout 0.0"
                                done
                            fi

                            # Dropout
                            for DROPOUT in 0.25; do
                                sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=2 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT"
                            done

                        done
                    done
                done
            done
        done
    done
done
