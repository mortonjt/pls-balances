# generate.py compositional_effect_size \
#     --max-alpha 100 --reps 100 --intervals 10 \
#     --n-species 100 --n-diff 50 \
#     --output-dir pls_balances/results/effect_size_benchmarks2

# generate.py compositional_variable_features \
#     --max-changing 500 --reps 50 --intervals 3 --n-species 100 \
#       --fold-change 2 \
#       --output-dir pls_balances/results/variable_features_benchmarks

generate.py compositional_regression_prefilter \
    --max-gradient 10 \
    --gradient-intervals 5 \
    --sigma 2 \
    --n-species 10 \
    --lam 0.1  \
    --max-contaminants 10 \
    --contaminant-intervals 3 \
    --output-dir pls_balances/results/variable_contaminants_benchmarks_test

generate.py compositional_regression_effect_size \
    --max-gradient 10 \
    --gradient-intervals 5 \
    --sigma 2 \
    --n-species 10 \
    --lam 0.1  \
    --max-beta 1 \
    --beta-intervals 3 \
    --output-dir pls_balances/results/effect_size_regression_benchmarks_test

generate.py compositional_effect_size \
    --max-alpha 10 --reps 10 --intervals 3 \
    --n-species 10 --n-diff 5 \
    --n-contaminants 3 --lam 0.1 \
    --output-dir pls_balances/results/effect_size_benchmarks_test

generate.py compositional_variable_features \
    --max-changing 5 --reps 10 --intervals 3 --n-species 10 \
    --fold-change 2 \
    --output-dir pls_balances/results/variable_features_benchmarks_test

snakemake --cores 8 --configfile variable_contaminants.yaml
snakemake --cores 8 --configfile effect_size.yaml
snakemake --cores 8 --configfile effect_size_regression.yaml
snakemake --cores 8 --configfile variable_features.yaml
