cd src

# for HIDDEN_DIMENSION in 1000 2500 5000 max
#     do
#     for CANCER_TYPE in blca all
#         do
#         for DATASET in rppa2gex cnapurity2gex avggexsubtype2gex
#             do
#             for NORMALIZE_OUTPUT in false true
#                 do
#                 bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --hidden_dimension $HIDDEN_DIMENSION --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers 0 --use_residual_connection false
#                 bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --hidden_dimension $HIDDEN_DIMENSION --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers 1 --use_residual_connection false
#                 bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type $CANCER_TYPE --dataset $DATASET --hidden_dimension $HIDDEN_DIMENSION --normalize_output $NORMALIZE_OUTPUT --num_hidden_layers 2 --use_residual_connection true
#                 done
#             done
#         done
#     done

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 1000 --normalize_output true --num_hidden_layers 1 --use_residual_connection false
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset rppa2gex --hidden_dimension 2500 --normalize_output true --num_hidden_layers 1 --use_residual_connection false

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 5000 --normalize_output true --num_hidden_layers 1 --use_residual_connection false
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset rppa2gex --hidden_dimension 1000 --normalize_output true --num_hidden_layers 0 --use_residual_connection false
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset avggexsubtype2gex --hidden_dimension 1000 --normalize_output true --num_hidden_layers 0 --use_residual_connection false
