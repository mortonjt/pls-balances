import os
import tempfile
import numpy as np
from itertools import product

# lam = 0.1
# category = 'n_diff'
# output_dir = 'test_data'
# n_contaminants = 1000
# intervals = 10


category = config['category']
benchmark = config['benchmark']
output_dir = config['output_dir']
intervals = config['intervals']
reps = config['reps']
sigma = config['sigma']

# lam = 0.1
# category = 'n_diff'
# output_dir = 'pls_balances/results/variable_features_benchmarks/'
# n_contaminants = 10
# intervals = 3
choice = 'abcdefghijklmnopqrstuvwxyz'
REPLICATES = list(choice[:reps])
SAMPLES = np.arange(intervals).astype(np.str)
SAMPLES = list(map(lambda x: '%s_%s' % x, product(SAMPLES, REPLICATES)))
TOOLS = config['tools']
#TOOLS = ['ancom', 'pls_balances', 't_test', 'mann_whitney']

rule all:
    input:
        output_dir+"confusion_matrix.summary"

rule inject_noise:
    input:
        table = output_dir + "table.{sample}.biom",
        metadata = output_dir + "metadata.{sample}.txt"
    output:
        output_dir + "table.noisy.{sample}.biom"
    run:
        shell("""
        generate.py noisify \
            --table-file {input.table} \
            --metadata-file {input.metadata} \
            --output-file {output}
        """)

rule run_tool:
    input:
        table = output_dir + "table.noisy.{sample}.biom",
        metadata = output_dir + "metadata.{sample}.txt"
    output:
        output_dir + "{tool}.{sample}.results"
    run:
        shell("""
        run.py {wildcards.tool}_cmd \
            --table-file {input.table} \
            --metadata-file {input.metadata} \
            --category {category} \
            --output-file {output}
        """)

rule summarize:
    input:
        tables = expand(output_dir + "table.noisy.{sample}.biom",
                        sample=SAMPLES),
        results = expand(output_dir + "{tool}.{sample}.results",
                         tool=TOOLS, sample=SAMPLES),
        truths = expand(output_dir + "truth.{sample}.csv",
                        sample=SAMPLES)
    output:
        output_dir + "{tool}.summary"
    run:
        from pls_balances.src.evaluate import compute_confusion_matrices
        compute_confusion_matrices(input.tables, input.results,
                                   input.truths, output[0])

rule aggregate_summaries:
    input:
        summaries = expand(output_dir + "{tool}.summary", tool=TOOLS),
        metadata = expand(output_dir + "metadata.{sample}.txt",
                          sample=SAMPLES),
        tables = expand(output_dir + "table.noisy.{sample}.biom",
                        sample=SAMPLES),
    output:
        output_dir + "confusion_matrix.summary"
    run:
        from pls_balances.src.evaluate import aggregate_summaries
        aggregate_summaries(input.summaries, input.tables, input.metadata,
                            benchmark, output[0])


