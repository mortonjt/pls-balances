# generate.py compositional_effect_size \
<<<<<<< HEAD
#     --max-alpha 4 --reps 200 --intervals 20 \
#     --n-species 10 --n-diff 50 \
#     --output-dir /home/mortonjt/Documents/pls-balances/pls_balances/results/effect_size_benchmarks_test
# 
=======
#     --max-alpha 100 --reps 100 --intervals 10 \
#     --n-species 100 --n-diff 50 \
#     --output-dir pls_balances/results/effect_size_benchmarks2

>>>>>>> 3584487daa76dbf5318d193a89cec78c864a9f28
# generate.py compositional_variable_features \
#     --max-changing 5 --reps 200 --intervals 21 --n-species 10 \
#     --fold-change 2 \
#     --output-dir pls_balances/results/variable_features_benchmarks_test

<<<<<<< HEAD
snakemake --cores 8 --configfile effect_size.yaml
snakemake --cores 8 --configfile variable_features.yaml
=======
generate.py compositional_regression_prefilter \
    --max-gradient 10 \
    --gradient-intervals 5 \
    --sigma 2 \
    --n-species 10 \
    --lam 0.1  \
    --max-contaminants 10 \
    --contaminant-intervals 3 \
    --output-dir pls_balances/results/variable_contaminants_benchmarks

# generate.py compositional_effect_size \
#     --max-alpha 10 --reps 10 --intervals 3 \
#     --n-species 10 --n-diff 5 \
#     --n-contaminants 3 --lam 0.1 \
#     --output-dir pls_balances/results/effect_size_benchmarks_test
snakemake --configfile variable_contaminants.yaml

# snakemake --cores 8 --configfile effect_size.yaml
# snakemake --cores 2 --configfile variable_features.yaml
>>>>>>> 3584487daa76dbf5318d193a89cec78c864a9f28
