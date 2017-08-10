# generate.py compositional_effect_size \
#     --max-alpha 2 --reps 30 --intervals 3 --n-species 20 \
#     --output-dir test_data

generate.py compositional_variable_features \
      --max-changing 500 --reps 100 --intervals 10 --n-species 1000 \
      --fold-change 2 \
      --output-dir test_data

snakemake
