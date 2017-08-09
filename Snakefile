import os
import tempfile
import numpy as np

SAMPLES = np.arange(3).astype(np.str)
TOOLS = ['ancom', 'pls_balances']


rule all:
    input:
        # expand("test_data/ancom.{sample}.results", sample=SAMPLES)
        #"test_data/ancom.summary"
        expand("test_data/{tool}.summary", tool=TOOLS)

rule inject_noise:
    input:
        table = "test_data/table.{sample}.biom",
        metadata = "test_data/metadata.{sample}.txt"
    output:
        "test_data/table.noisy.{sample}.biom"
    run:
        shell("""
        generate.py noisify \
            --lam 0.1 \
            --n-contaminants 30 \
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
        results = expand("test_data/{tool}.{sample}.results", sample=SAMPLES, tool=TOOLS),
        truths = expand("test_data/truth.{sample}.csv", sample=SAMPLES)
    output:
        "test_data/{tool}.summary"
    run:
        from pls_balances.src.evaluate import compute_confusion_matrices
        compute_confusion_matrices(input.tables, input.results,
                                   input.truths, output[0])

# rule aggregate_summaries:
#     input:
#         ancom='test_data/ancom.summary'
#         pls='test_data/pls.summary'

