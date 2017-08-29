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
    --contaminant-intervals 2 \
    --output-dir pls_balances/results/variable_contaminants

# snakemake --cores 8 --configfile effect_size.yaml
# snakemake --cores 2 --configfile variable_features.yaml
