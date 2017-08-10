generate.py compositional_effect_size \
    --max-alpha 10 --reps 30 --intervals 3 --n-species 20 \
    --output-dir pls_balances/results/effect_size_benchmarks

generate.py compositional_variable_features \
      --max-changing 5 --reps 30 --intervals 3 --n-species 10 \
      --fold-change 2 \
      --output-dir pls_balances/results/variable_features_benchmarks

snakemake --cores 2 --configfile effect_size.yaml
# snakemake --cores 2 --configfile variable_features.yaml
