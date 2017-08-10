generate.py compositional_effect_size \
    --max-alpha 1000 --reps 1000 --intervals 11 --n-species 1000 \
    --output-dir pls_balances/results/effect_size_benchmarks

generate.py compositional_variable_features \
      --max-changing 500 --reps 1000 --intervals 11 --n-species 1000 \
      --fold-change 2 \
      --output-dir pls_balances/results/variable_features_benchmarks

snakemake --cores 8 --configfile effect_size.yaml
snakemake --cores 8 --configfile variable_features.yaml
