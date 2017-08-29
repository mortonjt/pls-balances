# generate.py compositional_effect_size \
#     --max-alpha 4 --reps 200 --intervals 20 \
#     --n-species 10 --n-diff 50 \
#     --output-dir /home/mortonjt/Documents/pls-balances/pls_balances/results/effect_size_benchmarks_test
# 
# generate.py compositional_variable_features \
#     --max-changing 5 --reps 200 --intervals 21 --n-species 10 \
#     --fold-change 2 \
#     --output-dir pls_balances/results/variable_features_benchmarks_test

snakemake --cores 8 --configfile effect_size.yaml
snakemake --cores 8 --configfile variable_features.yaml
