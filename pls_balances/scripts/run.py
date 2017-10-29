import click
import numpy as np
import pandas as pd
from biom import load_table
from skbio.stats.composition import ancom
from skbio.stats.composition import (clr, centralize,
                                     multiplicative_replacement)
from sklearn.cross_decomposition import PLSRegression
from pls_balances.src.balances import round_balance
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr, dirichlet
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import tempfile
from subprocess import Popen
import io


@click.group()
def run():
    pass

@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def pls_balances_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)

    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))

    bootstraps = 100
    ls, rs, fs = [], [], []
    scores = pd.Series([0] * len(table.columns), table.columns)
    nums, denoms = set(table.columns), set(table.columns)
    for _ in range(bootstraps):
        _table = table.apply(lambda x: dirichlet.rvs(x+1).ravel(), axis=1)

        ctable = pd.DataFrame(clr(centralize(_table)),
                              index=table.index, columns=table.columns)

        rfc = PLSRegression(n_components=1)
        cats = np.unique(metadata[category])
        groups = (metadata[category] == cats[0]).astype(np.int)

        rfc.fit(X=ctable.values, Y=groups)

        pls_df = pd.DataFrame(rfc.x_weights_,
                              index=ctable.columns, columns=['PLS1'])
        l, r = round_balance(pls_df.values,
                             means_init=[[pls_df.PLS1.min()],
                                         [0],
                                         [pls_df.PLS1.max()]],
                             n_init=100)
        num = pls_df.loc[pls_df.PLS1 > r]
        denom = pls_df.loc[pls_df.PLS1 < l]
        nums = nums & set(num.index)
        denoms = denoms & set(denom.index)

    diff_features = scores.loc[scores > 10]

    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def ancom_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    res, _ = ancom(table+1, grouping=metadata[category])
    with open(output_file, 'w') as f:
        r = res["Reject null hypothesis"]
        f.write(','.join(res.loc[r].index.values))

@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def t_test_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cats = metadata[category]
    cs = np.unique(cats)
    def func(x):
        return ttest_ind(*[x[cats == k] for k in cs])
    m, p = np.apply_along_axis(func, axis=0,
                               arr=table.values)

    reject = p < 0.05
    features = pd.Series(reject, index=table.columns)
    diff_features = list(features.loc[features>0].index)
    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))

@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def mann_whitney_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cats = metadata[category]
    cs = np.unique(cats)
    def func(x):
        try: # catches the scenario where all values are the same.
            return mannwhitneyu(*[x[cats == k] for k in cs])
        except:
            return 0, 1

    m, p = np.apply_along_axis(func, axis=0,
                               arr=table.values)

    reject = p < 0.05
    features = pd.Series(reject, index=table.columns)
    diff_features = list(features.loc[features>0].index)
    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def lefse_cmd(table_file, metadata_file, category, output_file):
    fdir = tempfile.mkdtemp()
    pickle_file = '%s/tmp.pickle' % fdir
    tsv_file = '%s/tmp.txt' % fdir
    md_file = '%s/tmp.md.txt' % fdir
    lefse_file = '%s/tmp.lefse.txt' % fdir

    md = pd.read_table(metadata_file, index_col=0)
    md.to_csv(md_file, sep='\t', index_label='#SampleID')

    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()),
                         columns=table.ids(axis='sample'),
                         index=table.ids(axis='observation'))


    # add some garbage taxonomy column to make lefse happy
    table['Consensus Lineage'] = ['meh;blah'] * len(table.index)

    output = io.StringIO()
    output.write('# Constructed from biom file\n')
    table.to_csv(output, sep='\t', index_label='#OTU ID')
    open(tsv_file, 'w').write(output.getvalue())

    convert_cmd = ('source activate lefse; '
                   'qiime2lefse.py --in %s --md %s -c %s --out %s') % (
        tsv_file,
        md_file,
        category,
        lefse_file
        )

    format_cmd = ('source activate lefse;'
                  'lefse-format_input.py %s %s -c 1 -s -1 -u -1') % (
        lefse_file,
        pickle_file
        )

    lefse_cmd = ('source activate lefse; '
                 'run_lefse.py %s %s;'
                 'source activate pls-balances;') % (
        pickle_file,
        output_file
        )
    convert_proc = Popen(convert_cmd, shell=True)
    convert_proc.wait()
    format_proc = Popen(format_cmd, shell=True)
    format_proc.wait()
    lefse_proc = Popen(lefse_cmd, shell=True)
    lefse_proc.wait()
    res = pd.read_table(lefse_file, index_col=0, header=None)
    idx = res.iloc[:, 0] > 2
    diff_features = list(res.loc[res.iloc[:, 0] > 2].index)
    diff_features = list(map(lambda x: x.split('|')[-1], diff_features))
    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def pearson_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    gradient = metadata[category].astype(np.float)

    r, p = np.apply_along_axis(lambda x: pearsonr(x, gradient),
                               axis=0, arr=table.values)
    reject = np.logical_and(p < 0.05, r**2 > 0.5)
    features = pd.Series(reject, index=table.columns)
    diff_features = list(features.loc[features>0].index)
    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))

@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def spearman_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    gradient = metadata[category].astype(np.float)

    r, p = np.apply_along_axis(lambda x: spearmanr(x, gradient),
                               axis=0, arr=table.values)

    reject = np.logical_and(p < 0.05, r**2 > 0.5)
    features = pd.Series(reject, index=table.columns)
    diff_features = list(features.loc[features > 0].index)
    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def lasso_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cl = linear_model.LassoCV()

    cl.fit(X=table, y=metadata[category])
    idx = np.abs(cl.coef_) > 0
    diff_features = table.columns[idx]
    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))

@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def random_forest_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    rf=RandomForestRegressor()
    rf.fit(X=table, y=metadata[category])
    idx = np.abs(rf.feature_importances_) > 0
    diff_features = table.columns[idx]
    with open(output_file, 'w') as f:
        f.write(','.join(diff_features))


if __name__ == "__main__":
    run()
