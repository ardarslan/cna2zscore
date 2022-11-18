cd src
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset rppa2gex
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset rppa2gex
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset avggexsubtype2gex
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset avggexsubtype2gex

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 100
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 100

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 500
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 500

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 1000
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 1000

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 2000
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 2000

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 5000
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 5000


bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 100 --normalize_input true --normalize_output true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 100 --normalize_input true --normalize_output true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 500 --normalize_input true --normalize_output true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 500 --normalize_input true --normalize_output true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 1000 --normalize_input true --normalize_output true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 1000 --normalize_input true --normalize_output true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 2000 --normalize_input true --normalize_output true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 2000 --normalize_input true --normalize_output true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 5000 --normalize_input true --normalize_output true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 5000 --normalize_input true --normalize_output true


bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 100 --num_hidden_layers 2 --use_residual_connection true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 100 --num_hidden_layers 2 --use_residual_connection true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 500 --num_hidden_layers 2 --use_residual_connection true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 500 --num_hidden_layers 2 --use_residual_connection true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 1000 --num_hidden_layers 2 --use_residual_connection true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 1000 --num_hidden_layers 2 --use_residual_connection true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 2000 --num_hidden_layers 2 --use_residual_connection true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 2000 --num_hidden_layers 2 --use_residual_connection true

bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type all --dataset cnapurity2gex --hidden_dimension 5000 --num_hidden_layers 2 --use_residual_connection true
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python main.py --cancer_type blca --dataset cnapurity2gex --hidden_dimension 5000 --num_hidden_layers 2 --use_residual_connection true
