import os
import click
import pandas as pd
import numpy as np
from scipy.stats import expon
from skbio.stats.composition import perturb, closure
from pls_balances.src.generators import (compositional_effect_size_generator,
                                         compositional_variable_features_generator,
                                         compositional_regression_prefilter_generator,
                                         compositional_regression_effect_size_generator,
                                         library_size_difference_generator)

from pls_balances.src.sim import multinomial_sample, compositional_noise
from biom import Table, load_table
from biom.util import biom_open


def deposit(table, groups, truth,
            output_table, output_groups, output_truth):
    t = Table(table.T.values, table.columns.values, table.index.values)
    with biom_open(output_table, 'w') as f:
        t.to_hdf5(f, generated_by='moi')
    groups.to_csv(output_groups, sep='\t')
    with open(output_truth, 'w') as f:
        f.write(','.join(truth))


@click.group()
def generate():
    pass


@generate.command()
@click.option('--table-file',
              help='Input biom table of abundances.')
@click.option('--metadata-file',
              help='Input metadata file that was autogenerated.')
@click.option('--sigma', default=0.1,
              help='Variance of compositional noise')
@click.option('--output-file',
              help='output file of modified biom table.')
def noisify(table_file, metadata_file,
            sigma, output_file):

    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cov = np.eye(table.shape[1] - 1)
    m_noise = compositional_noise(cov, nsamp=table.shape[0])
    table_ = table.values
    table_ = np.vstack([
        perturb(table_[i, :], m_noise[i, :])
        for i in range(table_.shape[0])])

    # note that this assumes that the column is named `library_size
    table_ = pd.DataFrame(
        multinomial_sample(table_, depths=metadata['library_size']))
    table_.index = table.index
    table_.columns = list(table.columns)


    metadata['observed'] = np.sum(table_.sum(axis=0) > 0)
    metadata['unobserved'] = np.sum(table_.sum(axis=0)== 0)
    metadata.to_csv(metadata_file, sep='\t')

    # drop zeros -- they are not informative
    table_ = table_.loc[:, table_.sum(axis=0) > 0]
    t = Table(table_.T.values, table_.columns.values, table_.index.values)
    with biom_open(output_file, 'w') as f:
        t.to_hdf5(f, generated_by='moi')


@generate.command()
@click.option('--max-alpha', default=2,
              help='Maximum effect size.')
@click.option('--reps', default=30,
              help='Number of samples in each group.')
@click.option('--intervals', default=30,
              help='Number of effect size benchmarks to test.')
@click.option('--n-species', default=100,
              help='Number of species')
@click.option('--n-diff', default=50,
              help='Number of differentially abundant species')
@click.option('--n-contaminants', default=100,
              help='Number of species')
@click.option('--lam', default=0.1,
              help='Scale factor for exponential contamination urn.')
@click.option('--library-size', default=10000,
              help='Library size (i.e. sequencing depth)..')
@click.option('--asymmetry', is_flag=True, default=False,
              help=('Fold-change applied to max-changing species in both '
                    'sample groups = False'))
@click.option('--fold-balance', is_flag=True, default=False,
              help=('If False, then only the max-changing species will be '
                    'increased in both sample groups.  Otherwise, one of the '
                    'groups will be decreased, while the other group increases.'))
@click.option('--template-biom', default=None,
              help='Template biom file path.')
@click.option('--template-sample-name', default=None,
              help='Template sample name.')
@click.option('--output-dir',
              help='output directory')
def compositional_effect_size(max_alpha, reps, intervals,
                              n_species, n_diff,
                              n_contaminants, lam,
                              library_size,
                              asymmetry, fold_balance,
                              template_biom,
                              template_sample_name,
                              output_dir):
    if template_biom is not None:
        templ = load_table(template_biom)
        template = templ.data(id=template_sample_name, axis='sample')
    else:
        template = None

    os.mkdir(output_dir)
    gen = compositional_effect_size_generator(
        max_alpha, reps, intervals, n_species, n_diff,
        n_contaminants, lam, library_size=library_size,
        asymmetry=asymmetry, fold_balance=fold_balance,
        template=template
    )

    for i, g in enumerate(gen):
        table, groups, truth = g
        output_table = "%s/table.%d.biom" % (output_dir, i)
        output_groups = "%s/metadata.%d.txt" % (output_dir, i)
        output_truth = "%s/truth.%d.csv" % (output_dir, i)
        deposit(table, groups, truth,
                output_table, output_groups, output_truth)

@generate.command()
@click.option('--effect-size', default=2,
              help=('The effect size difference between the '
                    'differientially abundant features.'))
@click.option('--reps', default=30,
              help='Number of samples in each group.')
@click.option('--intervals', default=30,
              help='Number of effect size benchmarks to test.')
@click.option('--n-species', default=100,
              help='Number of species')
@click.option('--n-diff', default=50,
              help='Number of differentially abundant species')
@click.option('--lam-diff', default=0.1,
              help='Scale factor for exponential contamination urn ')
@click.option('--n-contaminants', default=100,
              help='Number of species')
@click.option('--lam-contaminants', default=0.1,
              help='Scale factor for the differientially abundant species')
@click.option('--min-library-size', default=1000,
              help='Library size (i.e. sequencing depth)..')
@click.option('--max-library-size', default=10000,
              help='Library size (i.e. sequencing depth)..')
@click.option('--output-dir',
              help='output directory')
def library_size_difference(effect_size, reps, intervals,
                            n_species, n_diff, lam_diff,
                            n_contaminants, lam_contaminants,
                            min_library_size, max_library_size,
                            output_dir):

    gen = library_size_difference_generator(
        effect_size=effect_size,
        reps=reps,
        intervals=intervals,
        n_species=n_species,
        n_diff=n_diff,
        lam_diff=lam_diff,
        n_contaminants=n_contaminants,
        lam_contaminants=lam_contaminants,
        min_library_size=min_library_size,
        max_library_size=max_library_size)

    os.mkdir(output_dir)
    for i, g in enumerate(gen):
        table, groups, truth = g
        output_table = "%s/table.%d.biom" % (output_dir, i)
        output_groups = "%s/metadata.%d.txt" % (output_dir, i)
        output_truth = "%s/truth.%d.csv" % (output_dir, i)
        deposit(table, groups, truth,
                output_table, output_groups, output_truth)


@generate.command()
@click.option('--max-changing', default=2,
              help='Maximum number of changing species.')
@click.option('--fold-change', default=2,
              help='Fold change of changing species.')
@click.option('--reps', default=30,
              help='Number of samples in each group.')
@click.option('--intervals', default=30,
              help='Number of effect size benchmarks to test.')
@click.option('--n-species', default=100,
              help='Number of species')
@click.option('--library-size', default=10000,
              help='Library size (i.e. sequencing depth)..')
@click.option('--asymmetry', is_flag=True, default=False,
              help=('Fold-change applied to max-changing species in both '
                    'sample groups = False'))
@click.option('--fold-balance', is_flag=True, default=False,
              help=('If False, then only the max-changing species will be '
                    'increased in both sample groups.  Otherwise, one of the '
                    'groups will be decreased, while the other group increases.'))
@click.option('--n-contaminants', default=100,
              help='Number of species')
@click.option('--lam', default=0.1,
              help='Scale factor for exponential contamination urn.')
@click.option('--template-biom', default=None,
              help='Template biom file path.')
@click.option('--template-sample-name', default=None,
              help='Template sample name.')
@click.option('--output-dir',
              help='output directory')
def compositional_variable_features(max_changing, fold_change, reps,
                                    intervals, n_species,
                                    library_size,
                                    asymmetry, fold_balance,
                                    n_contaminants, lam,
                                    template_biom,
                                    template_sample_name,
                                    output_dir):

    if template_biom is not None:
        templ = load_table(template_biom)
        template = templ.data(id=template_sample_name, axis='sample')
    else:
        template = None

    gen = compositional_variable_features_generator(
        max_changing=max_changing, fold_change=fold_change,
        library_size=library_size,
        reps=reps, intervals=intervals, n_species=n_species,
        n_contaminants=n_contaminants, lam=lam, template=template,
        asymmetry=asymmetry, fold_balance=fold_balance
    )

    os.mkdir(output_dir)
    for i, g in enumerate(gen):
        table, groups, truth = g
        output_table = "%s/table.%d.biom" % (output_dir, i)
        output_groups = "%s/metadata.%d.txt" % (output_dir, i)
        output_truth = "%s/truth.%d.csv" % (output_dir, i)
        deposit(table, groups, truth,
                output_table, output_groups, output_truth)


@generate.command()
@click.option('--max-gradient', default=2,
              help='Maximum value of the gradient (gradient starts at zero).')
@click.option('--gradient-intervals', default=2,
              help='Number of intervals within the gradient.')
@click.option('--sigma', default=3,
              help='Variance of species normal distribution along gradient.')
@click.option('--n-species', default=100,
              help='Number of species')
@click.option('--lam', default=0.1,
              help='Scale factor for exponential contamination urn.')
@click.option('--max-contaminants', default=100,
              help='Maximum number of contaminants in urn.')
@click.option('--contaminant-intervals', default=100,
              help='Number intervals for varying number of contaminant species.')
@click.option('--output-dir',
              help='output directory')
def compositional_regression_prefilter(max_gradient,
                                       gradient_intervals,
                                       sigma,
                                       n_species,
                                       lam,
                                       max_contaminants,
                                       contaminant_intervals,
                                       output_dir):

    gen = compositional_regression_prefilter_generator(
        max_gradient, gradient_intervals, sigma,
        n_species, lam, max_contaminants,
        contaminant_intervals
    )
    os.mkdir(output_dir)
    for i, g in enumerate(gen):
        table, groups, truth = g
        output_table = "%s/table.%d.biom" % (output_dir, i)
        output_groups = "%s/metadata.%d.txt" % (output_dir, i)
        output_truth = "%s/truth.%d.csv" % (output_dir, i)
        deposit(table, groups, truth,
                output_table, output_groups, output_truth)


@generate.command()
@click.option('--max-gradient', default=2,
              help='Maximum value of the gradient (gradient starts at zero).')
@click.option('--gradient-intervals', default=2,
              help='Number of intervals within the gradient.')
@click.option('--sigma', default=3,
              help='Variance of species normal distribution along gradient.')
@click.option('--n-species', default=100,
              help='Number of species')
@click.option('--n-contaminants', default=100,
              help='Number of species')
@click.option('--lam', default=0.1,
              help='Scale factor for exponential contamination urn.')
@click.option('--max-beta', default=100,
              help='Maximum number of contaminants in urn.')
@click.option('--beta-intervals', default=100,
              help='Number intervals for varying number of contaminant species.')
@click.option('--output-dir',
              help='output directory')
def compositional_regression_effect_size(max_gradient,
                                         gradient_intervals,
                                         sigma,
                                         n_species,
                                         n_contaminants,
                                         lam,
                                         max_beta,
                                         beta_intervals,
                                         output_dir):

    gen = compositional_regression_effect_size_generator(max_gradient,
                                                         gradient_intervals,
                                                         sigma,
                                                         n_species,
                                                         n_contaminants,
                                                         lam,
                                                         max_beta,
                                                         beta_intervals)

    os.mkdir(output_dir)
    for i, g in enumerate(gen):
        table, groups, truth = g
        output_table = "%s/table.%d.biom" % (output_dir, i)
        output_groups = "%s/metadata.%d.txt" % (output_dir, i)
        output_truth = "%s/truth.%d.csv" % (output_dir, i)
        deposit(table, groups, truth,
                output_table, output_groups, output_truth)


if __name__ == "__main__":
    generate()
