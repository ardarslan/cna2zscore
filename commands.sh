cd src

NUM_JOBS=0


sleep_if_necessary() {
    if [ $(expr $(($NUM_JOBS+1)) % 100) == "0" ]; then
        read -p "Number of submitted jobs: $NUM_JOBS. Sleeping for 3 seconds..." -t 3
    fi
}

for CANCER_TYPE in 'all'; do
    for DATASET in 'rppa2zscore' 'unthresholdedcnapurity2zscore'; do
        for MODEL in 'sklearn_per_gene' 'sklearn_linear'; do
            if [[ $DATASET = 'rppa2zscore' ]]; then
                GENE_TYPE_OPTIONS=("rppa_genes")
            elif [[ $MODEL = 'sklearn_per_gene' || $MODEL = 'sklearn_linear' ]]; then
                GENE_TYPE_OPTIONS=("168_highly_expressed_genes" "1000_highly_expressed_genes" "rppa_genes" "all_genes")
            else
                GENE_TYPE_OPTIONS=("168_highly_expressed_genes" "1000_highly_expressed_genes" "rppa_genes" "all_genes")
            fi

            if [[ $MODEL = 'sklearn_per_gene' || $MODEL = 'sklearn_linear' ]]; then
                DROPOUT_OPTIONS=(0.00)
                LEARNING_RATE_OPTIONS=(0.0)
                MAIN_FILE_NAME=("main_sklearn.py")
                GPU_SETTINGS=("")
            else
                DROPOUT_OPTIONS=(0.00 0.25 0.33 0.50)
                LEARNING_RATE_OPTIONS=(0.001 0.01)
                MAIN_FILE_NAME=("main_dl.py")
                GPU_SETTINGS=("--gpus=1 --gres=gpumem:12288 ")
            fi

            if [[ $MODEL = 'sklearn_per_gene' || $MODEL = 'dl_per_gene' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                L1_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0)
                L2_REG_COEFF_OPTIONS=(0.0)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
                PER_CHROMOSOME_OPTIONS=(false)
            elif [[ $MODEL = 'gene_embeddings' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                L1_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                L2_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                GENE_EMBEDDING_SIZE_OPTIONS=(4 16 64)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
                PER_CHROMOSOME_OPTIONS=(false true)
            elif [[ $MODEL = 'sklearn_linear' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                L1_REG_DIAGONAL_COEFF_OPTIONS=(0.0)
                L2_REG_DIAGONAL_COEFF_OPTIONS=(0.0)
                L1_REG_NONDIAGONAL_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                L2_REG_NONDIAGONAL_COEFF_OPTIONS=(0.0)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
                PER_CHROMOSOME_OPTIONS=(false true)
            elif [[ $MODEL = 'dl_linear' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.0)
                NUM_NONLINEAR_LAYERS_OPTIONS=(0)
                L1_REG_DIAGONAL_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                L2_REG_DIAGONAL_COEFF_OPTIONS=(0.0)
                L1_REG_NONDIAGONAL_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                L2_REG_NONDIAGONAL_COEFF_OPTIONS=(0.0)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
                PER_CHROMOSOME_OPTIONS=(false true)
            elif [[ $MODEL = 'mlp' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.10 0.25 0.50)
                NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                L1_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                L2_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
                PER_CHROMOSOME_OPTIONS=(false true)
            elif [[ $MODEL = 'rescon_mlp' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false true)
                HIDDEN_DIMENSION_OPTIONS=(0.10 0.25 0.50)
                NUM_NONLINEAR_LAYERS_OPTIONS=(1 2)
                L1_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                L2_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                GENE_EMBEDDING_SIZE_OPTIONS=(0)
                NUM_ATTENTION_HEADS_OPTIONS=(0)
                PER_CHROMOSOME_OPTIONS=(false true)
            elif [[ $MODEL = 'transformer' ]]; then
                RESCON_DIAGONAL_W_OPTIONS=(false)
                HIDDEN_DIMENSION_OPTIONS=(0.10 0.25 0.50)
                NUM_NONLINEAR_LAYERS_OPTIONS=(0 1 2)
                L1_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                L2_REG_COEFF_OPTIONS=(0.0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01)
                GENE_EMBEDDING_SIZE_OPTIONS=(4 16 64)
                NUM_ATTENTION_HEADS_OPTIONS=(4)
                PER_CHROMOSOME_OPTIONS=(false true)
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
                                    for DROPOUT in "${DROPOUT_OPTIONS[@]}"; do
                                        for LEARNING_RATE in "${LEARNING_RATE_OPTIONS[@]}"; do
                                            for PER_CHROMOSOME in "${PER_CHROMOSOME_OPTIONS[@]}"; do

                                                # No regularization
                                                # sleep_if_necessary
                                                # sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 $GPU_SETTINGS --wrap="python $MAIN_FILE_NAME --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT --per_chromosome $PER_CHROMOSOME"
                                                # let NUM_JOBS=NUM_JOBS+1

                                                # L1 regularization
                                                if [[ $MODEL = "dl_linear" ]]; then
                                                    for L1_REG_DIAGONAL_COEFF in "${L1_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                                        for L1_REG_NONDIAGONAL_COEFF in "${L1_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                                            if (( $(echo "$L1_REG_DIAGONAL_COEFF > $L1_REG_NONDIAGONAL_COEFF" |bc -l) )); then
                                                                continue
                                                            else
                                                                sleep_if_necessary
                                                                sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 $GPU_SETTINGS --wrap="python $MAIN_FILE_NAME --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff $L1_REG_DIAGONAL_COEFF --l1_reg_nondiagonal_coeff $L1_REG_NONDIAGONAL_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT --per_chromosome $PER_CHROMOSOME"
                                                                let NUM_JOBS=NUM_JOBS+1
                                                            fi
                                                        done
                                                    done
                                                else
                                                    for L1_REG_COEFF in "${L1_REG_COEFF_OPTIONS}"; do
                                                        sleep_if_necessary
                                                        sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 $GPU_SETTINGS --wrap="python $MAIN_FILE_NAME --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff $L1_REG_COEFF --l1_reg_nondiagonal_coeff $L1_REG_COEFF --l2_reg_diagonal_coeff 0.0 --l2_reg_nondiagonal_coeff 0.0 --dropout $DROPOUT --per_chromosome $PER_CHROMOSOME"
                                                        let NUM_JOBS=NUM_JOBS+1
                                                    done
                                                fi

                                                # L2 regularization
                                                # if [[ $MODEL = "dl_linear" ]]; then
                                                #     for L2_REG_DIAGONAL_COEFF in "${L2_REG_DIAGONAL_COEFF_OPTIONS[@]}"; do
                                                #         for L2_REG_NONDIAGONAL_COEFF in "${L2_REG_NONDIAGONAL_COEFF_OPTIONS[@]}"; do
                                                #             if (( $(echo "$L2_REG_DIAGONAL_COEFF > $L2_REG_NONDIAGONAL_COEFF" |bc -l) )); then
                                                #                 continue
                                                #             else
                                                #                 sleep_if_necessary
                                                #                 sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 $GPU_SETTINGS --wrap="python $MAIN_FILE_NAME --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_DIAGONAL_COEFF --l2_reg_nondiagonal_coeff $L2_REG_NONDIAGONAL_COEFF --dropout $DROPOUT --per_chromosome $PER_CHROMOSOME"
                                                #                 let NUM_JOBS=NUM_JOBS+1
                                                #             fi
                                                #         done
                                                #     done
                                                # else
                                                #     for L2_REG_COEFF in "${L2_REG_COEFF_OPTIONS[@]}"; do
                                                #         sleep_if_necessary
                                                #         sbatch --time=1440 --ntasks=2 --mem-per-cpu=32768 $GPU_SETTINGS --wrap="python $MAIN_FILE_NAME --dataset $DATASET --cancer_type $CANCER_TYPE --gene_type $GENE_TYPE --model $MODEL --rescon_diagonal_W $RESCON_DIAGONAL_W --gene_embedding_size $GENE_EMBEDDING_SIZE --num_attention_heads $NUM_ATTENTION_HEADS --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --hidden_dimension $HIDDEN_DIMENSION --learning_rate $LEARNING_RATE --l1_reg_diagonal_coeff 0.0 --l1_reg_nondiagonal_coeff 0.0 --l2_reg_diagonal_coeff $L2_REG_COEFF --l2_reg_nondiagonal_coeff $L2_REG_COEFF --dropout $DROPOUT --per_chromosome $PER_CHROMOSOME"
                                                #         let NUM_JOBS=NUM_JOBS+1
                                                #     done
                                                # fi

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
done

echo "Number of submitted jobs: $NUM_JOBS."
