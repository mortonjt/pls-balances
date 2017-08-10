import os
import tempfile
import numpy as np

lam = 0.1
category = 'effect_size'
output_dir = '{output_dir}'
n_contaminants = 30

SAMPLES = np.arange(3).astype(np.str)
TOOLS = ['ancom', 'pls_balances']

# question : how can I get rid of test_data?

rule all:
    input:
        # expand("{output_dir}/ancom.{sample}.results", sample=SAMPLES)
        #"{output_dir}/ancom.summary"
        # expand("{output_dir}/{tool}.summary", tool=TOOLS)
        "test_data/confusion_matrix.summary"

rule inject_noise:
    input:
        table = "test_data/table.{sample}.biom",
        metadata = "test_data/metadata.{sample}.txt"
    output:
        "test_data/table.noisy.{sample}.biom"
    run:
        shell("""
        generate.py noisify \
            --lam {lam} \
            --n-contaminants {n_contaminants} \
            --table-file {input.table} \
            --metadata-file {input.metadata} \
            --output-file {output}
        """)

rule run_tool:
    input:
        table = "test_data/table.noisy.{sample}.biom",
        metadata = "test_data/metadata.{sample}.txt"
    output:
        "test_data/{tool}.{sample}.results"
    run:
        shell("""
        run.py {wildcards.tool}_cmd \
            --table-file {input.table} \
            --metadata-file {input.metadata} \
            --category group \
            --output-file {output}
        """)

rule summarize:
    input:
        tables = expand("test_data/table.noisy.{sample}.biom", sample=SAMPLES),
        results = expand("test_data/{tool}.{sample}.results",
                         tool=TOOLS, sample=SAMPLES),
        truths = expand("test_data/truth.{sample}.csv", sample=SAMPLES)
    output:
        "test_data/{tool}.summary"
    run:
        from pls_balances.src.evaluate import compute_confusion_matrices
        compute_confusion_matrices(input.tables, input.results,
                                   input.truths, output[0])

rule aggregate_summaries:
    input:
        summaries = expand("test_data/{tool}.summary", tool=TOOLS),
        metadata = expand("test_data/metadata.{sample}.txt", sample=SAMPLES),
        tables = expand("test_data/table.noisy.{sample}.biom", sample=SAMPLES),
    output:
        "test_data/confusion_matrix.summary"
    run:
        from pls_balances.src.evaluate import aggregate_summaries
        aggregate_summaries(input.summaries, input.tables, input.metadata,
                            category, output[0])


