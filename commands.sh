cd src

NUM_JOBS=0

for CANCER_TYPE in 'all'; do
    for DATASET in 'unthresholdedcnapurity2gex' 'rppa2gex'; do
        if [[ $DATASET = 'rppa2gex' ]]; then
            declare -a MODEL_OPTIONS=("linear" "linear_per_chromosome_24") #  "mlp" "transformer")
        elif [[ $DATASET = 'unthresholdedcnapurity2gex' ]]; then
            declare -a MODEL_OPTIONS=("linear" "linear_per_chromosome_24") # "mlp" "mlp_per_chromosome_24" "transformer")
        else
            echo "DATASET is not a valid $DATASET."
            exit 1
        fi

        for MODEL in "${MODEL_OPTIONS[@]}"; do
            if [[ $DATASET = 'rppa2gex' ]]; then
                declare -a GENE_TYPE_OPTIONS=("rppa_genes")
            elif [[ $MODEL = 'linear_per_chromosome_24' || $MODEL = 'mlp_per_chromosome_24' ]]; then
                declare -a GENE_TYPE_OPTIONS=("chromosome_24_genes")
            elif [[ $MODEL = 'linear_per_chromosome_all' || $MODEL = 'mlp_per_chromosome_all' ]]; then
                declare -a GENE_TYPE_OPTIONS=("chromosome_all_genes")
            else
                declare -a GENE_TYPE_OPTIONS=("1000_highly_expressed_genes" "rppa_genes" "chromosome_24_genes")
            fi

            if [[ $MODEL = 'linear' || $MODEL = 'linear_per_chromosome_all' || $MODEL = 'linear_per_chromosome_24' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                L1_REG_DIAGONAL_COEFF_OPTIONS=(0.1)
                L2_REG_DIAGONAL_COEFF_OPTIONS=(0.1)
                L1_REG_NONDIAGONAL_COEFF_OPTIONS=(0.1)
                L2_REG_NONDIAGONAL_COEFF_OPTIONS=(0.1)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
            elif [[ $MODEL = 'mlp' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.10 0.25 0.5 1.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                L1_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                L2_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
            elif [[ $MODEL = 'mlp_per_chromosome_all' || $MODEL = 'mlp_per_chromosome_24' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.10 0.25 0.5 1.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                L1_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                L2_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
            elif [[ $MODEL = 'rescon_mlp' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false true)
                HIDDEN_DIMENSION_OPTIONS=(0.10 0.25 0.5 1.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                L1_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                L2_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
            elif [[ $MODEL = 'transformer' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                L1_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                L2_REG_COEFF_OPTIONS=(0.00001 0.0001 0.001 0.01 0.1)
                GENE_EMBEDDING_SIZE_OPTIONS=(4 16 64)
                NUM_ATTENTION_HEADS_OPTIONS=(4 8)
            else
                echo "MODEL is not a valid $MODEL."
                exit 1
            fi

            for GENE_TYPE in "${GENE_TYPE_OPTIONS[@]}"; do
                for RESCON_DIAGONAL_W in "${RESCON_DIAGONAL_W_OPTIONS[@]}"; do
                    for GENE_EMBEDDING_SIZE in "${GENE_EMBEDDING_SIZE_OPTIONS[@]}"; do
                        for NUM_ATTENTION_HEADS in "${NUM_ATTENTION_HEADS_OPTIONS[@]}"; do
                            for NUM_NONLINEAR_LAYERS in "${NUM_NONLINEAR_LAYERS_OPTIONS[@]}"; do
                                for HIDDEN_DIMENSION in "${HIDDEN_DIMENSION_OPTIONS[@]}"; do
                                    for DROPOUT in 0.0 0.25 0.33; do
                                        for LEARNING_RATE in 0.000001 0.00001 0.0001; do

                                            # No regularization
                                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT"
                                            let NUM_JOBS=NUM_JOBS+1

                                            # L1 regularization
                                            if [[ $MODEL = "linear" || $MODEL = "linear_per_chromosome_all" || $MODEL = "linear_per_chromosome_24" ]]; then
                                                for L1_REG_DIAGONAL_COEFF in "${L1_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                                    for L1_REG_NONDIAGONAL_COEFF in "${L1_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                                        if (( $(echo "$L1_REG_DIAGONAL_COEFF > $L1_REG_NONDIAGONAL_COEFF" |bc -l) )); then
                                                            continue
                                                        else
                                                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff $L1_REG_DIAGONAL_COEFF --l1_reg_nondiagonal_coeff $L1_REG_NONDIAGONAL_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT"
                                                            let NUM_JOBS=NUM_JOBS+1
                                                        fi
                                                    done
                                                done
                                            else
                                                for L1_REG_COEFF in "${L1_REG_COEFF_OPTIONS}"; do
                                                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff $L1_REG_COEFF --l1_reg_nondiagonal_coeff $L1_REG_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT"
                                                    let NUM_JOBS=NUM_JOBS+1
                                                done
                                            fi

                                            # L2 regularization
                                            if [[ $MODEL = "linear" || $MODEL = "linear_per_chromosome_all" || $MODEL = "linear_per_chromosome_24" ]]; then
                                                for L2_REG_DIAGONAL_COEFF in "${L2_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                                    for L2_REG_NONDIAGONAL_COEFF in "${L2_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                                        if (( $(echo "$L2_REG_DIAGONAL_COEFF > $L2_REG_NONDIAGONAL_COEFF" |bc -l) )); then
                                                            continue
                                                        else
                                                            sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_DIAGONAL_COEFF --l2_reg_nondiagonal_coeff $L2_REG_NONDIAGONAL_COEFF --dropout $DROPOUT"
                                                            let NUM_JOBS=NUM_JOBS+1
                                                        fi
                                                    done
                                                done
                                            else
                                                for L2_REG_COEFF in "${L2_REG_COEFF_OPTIONS[@]}"; do
                                                    sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 --gpus=1 --gres=gpumem:12288 --wrap="python main.py --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_COEFF --l2_reg_nondiagonal_coeff $L2_REG_COEFF --dropout $DROPOUT"
                                                    let NUM_JOBS=NUM_JOBS+1
                                                done
                                            fi

                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Number of jobs is $NUM_JOBS."
